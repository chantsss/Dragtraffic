import copy
import logging
import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from dragtraffic.act.utils.model_utils import MLP_3, CG_stacked
from dragtraffic.diffusion.diffusion import VarianceSchedule, TransformerConcatLinear
from dragtraffic.diffusion.openaimodel import UNetModel
from dragtraffic.act_dataset import NORM_RANGE, NORM_SPEED

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0

class actuatordiff(pl.LightningModule):
    """ A transformer model with wider latent space """
    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        # input embedding stem
        hidden_dim = cfg['embed_dim'] if 'embed_dim' in cfg else 256
        self.CG_agent = CG_stacked(5, hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.feature_dim = 2
        self.num_modes = self.cfg['num_modes'] if 'num_modes' in self.cfg else 6

        hidden_dim_condition_context = 0
        if self.cfg['condition_context']:
            hidden_dim_condition_context = hidden_dim
            self.condition_context_encode = MLP_3([8, int(hidden_dim_condition_context/4), int(hidden_dim_condition_context/2), hidden_dim_condition_context])
            self.feature_dim = 2

        self.CG_all = CG_stacked(5, hidden_dim * self.feature_dim + hidden_dim_condition_context)
        self.agent_encode = MLP_3([8, 256, 512, hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)

        self.pred_len = 89
        self.apply(self._init_weights)

        self.diffusion_context_size = hidden_dim * self.feature_dim + hidden_dim_condition_context
        self.nhead = 4
        self.tf_layer = 3

        self.diffusion_training_step = cfg['diffusion_training_step'] if 'diffusion_training_step' in cfg else 100
        self.diffusion_beta_T = 5e-2
        self.ddim_sample_step = cfg['diffusion_sample_step'] if 'diffusion_sample_step'  in cfg else 100
        self.ddim_step = int(100)
        self.point_dim = 5

        self.var_sched = VarianceSchedule(
                num_steps=self.diffusion_training_step,
                beta_T=self.diffusion_beta_T,
                mode='linear')
        
        if self.cfg['diff_base'] == 'transformer':
            self.diffnet = TransformerConcatLinear(point_dim=self.point_dim, 
                                                context_dim=self.diffusion_context_size,
                                                nhead=self.nhead, 
                                                tf_layer=self.tf_layer, 
                                                residual=False)
        else:
            self.diffnet = UNetModel(
                image_size=64,
                in_channels=self.point_dim,
                model_channels=64,
                out_channels=self.point_dim,
                num_res_blocks=2,
                attention_resolutions=(4, 2),
                dropout=0,
                channel_mult=(1, 2, 4),
                conv_resample=True,
                dims=1,
                num_classes=None,
                use_checkpoint=False,
                use_fp16=False,
                num_heads=-1,
                num_head_channels=32,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
                use_spatial_transformer=False,
                transformer_depth=1,
                context_dim=None,
                n_embed=None,
                legacy=True,
            )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        pred_embed = self.encode(batch)
        pred = self.decode(pred_embed, batch)
        pred['velo'] *= NORM_SPEED
        pred['pos'] *= NORM_SPEED
        return pred

    def training_step(self, batch, batch_idx):
        pred_embed = self.encode(batch)
        pred = None
        loss, loss_dict = self.loss(pred, pred_embed, batch)
        loss_dict = {'train/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_embed = self.encode(batch)
        loss, loss_dict = self.loss(None, pred_embed, batch)
        loss_dict = {'val/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def test_step(self, batch, sample_num=None, vis=False):
        with torch.no_grad():
            if sample_num is None:
                sample_num = self.num_modes
            pred_embed = self.encode(batch, vis)
            pred = self.decode(pred_embed, batch, num_samples=sample_num)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.2, verbose=True)
        return [optimizer], [scheduler]

    def loss(self, pred, pred_embed, gt, t=None):
        x_0 = (gt['gt_pos'][:, 1:] - gt['gt_pos'][:, :-1]).cuda()
        v_0 = gt['gt_vel'][:, 1:, :].cuda() 
        heading_0 = (gt['gt_heading'][:, 1:] - gt['gt_heading'][:, :-1]).cuda().unsqueeze(-1)
        xyh_0 = torch.cat([x_0, v_0, heading_0], -1)
        batch_size, _, point_dim = xyh_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()
        e_rand = torch.randn_like(xyh_0).cuda()
        context = pred_embed
        e_theta = self.diffnet(c0 * xyh_0 + c1 * e_rand, beta, context)
        diffusion_loss = F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.reshape(-1, point_dim), reduction='none')
        diffusion_loss[..., :2] = diffusion_loss[..., :2] * NORM_RANGE
        diffusion_loss[..., 2:4] = diffusion_loss[..., 2:4] * NORM_SPEED
        diffusion_loss = diffusion_loss.mean()
        loss_sum = diffusion_loss
        loss_dict = {}
        loss_dict['diffusion_loss'] = diffusion_loss
        return loss_sum, loss_dict

    def encode(self, data, not_from_drag=False):
        agent = data['agent']
        agent_mask = data['agent_mask']
        if self.cfg['condition_context']:
            goal_condition_indices = (data['gt_pos'].shape[1] -1) *\
                  torch.ones(data['gt_pos'].shape[0], dtype=torch.long).to(data['gt_pos'].device)
            if 'goal_condition_indices' in data:
                goal_condition_indices = data['goal_condition_indices'].long()
            gt_pos = torch.gather(data['gt_pos'], 1, goal_condition_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            gt_vel = torch.gather(data['gt_vel'], 1, goal_condition_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            gt_heading = torch.gather(data['gt_heading'], 1, goal_condition_indices.view(-1, 1))
            type_l_w = data['agent'][:, 0, 5:8]
            condition_context = torch.cat([gt_pos, gt_vel, gt_heading, type_l_w], dim=-1)
        all_vec = torch.cat([data['center'], data['cross'], data['bound']], dim=-2)
        line_mask = torch.cat([data['center_mask'], data['cross_mask'], data['bound_mask']], dim=1)
        polyline = all_vec[..., :4]
        polyline_type = all_vec[..., 4].to(int)
        polyline_traf = all_vec[..., 5].to(int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        agent_enc = self.agent_encode(agent)
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        b, a, d = agent_enc.shape
        device = agent_enc.device
        context_agent = torch.ones([b, d]).to(device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        if self.cfg['condition_context']:
            if len(condition_context.shape) == 3:
                condition_context = condition_context.squeeze(1)
            condition_context = self.condition_context_encode(condition_context)
            all_context = torch.cat([condition_context, context_agent, context_line], dim=-1)
        else:
            all_context = torch.cat([context_agent, context_line], dim=-1)
        pred_embed = all_context
        return pred_embed

    def decode(self, pred_embed, gt, dt=0.1, num_samples=None):
        pred = self.sample_batch(num_samples, self.pred_len, pred_embed, point_dim=self.point_dim)
        pos_pred = pred[..., :2].cumsum(-2)
        velo_pred = pred[..., 2:4]
        heading_pred = pred[..., -1].cumsum(-1)
        pred = {}
        pred['velo'] = velo_pred
        pred['pos'] = pos_pred
        pred['heading'] = heading_pred
        return pred

    def sample(self, num_samples, num_points, all_context, point_dim=2, ddim_step=200):
        batch_size = all_context.size(0)
        all_outputs = []
        for k in range(num_samples):
            context = all_context[:, k]
            x_T = torch.randn([batch_size, num_points, point_dim]).to(all_context.device)
            ts = np.linspace(self.var_sched.num_steps, 0, (self.ddim_sample_step + 1))
            simple_var = False
            eta = 0
            if simple_var:
                eta = 1
            x_t = x_T
            for i in range(1,self.ddim_sample_step+1):
                cur_t = int(ts[i - 1])
                prev_t = int(ts[i])
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t]*batch_size]
                eps = self.diffnet(x_t, beta, context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)
                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term
        all_outputs.append(x_t)
        all_outputs = torch.stack(all_outputs)
        all_outputs = all_outputs.permute(1, 0, 2, 3)
        return all_outputs

    def sample_batch(self, num_samples, num_points, all_context, point_dim=2, ddim_step=200):
        batch_size = all_context.shape[0]
        all_context = all_context.repeat(num_samples, 1)
        all_outputs = []
        context = all_context.squeeze(1)
        x_T = torch.randn([batch_size*num_samples, num_points, point_dim]).to(all_context.device)
        ts = np.linspace(self.var_sched.num_steps, 0, (self.ddim_sample_step + 1))
        simple_var = False
        eta = 0
        if simple_var:
            eta = 1
        x_t = x_T
        for i in range(1,self.ddim_sample_step+1):
            cur_t = int(ts[i - 1])
            prev_t = int(ts[i])
            ab_cur = self.var_sched.alpha_bars[cur_t]
            ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
            beta = self.var_sched.betas[[cur_t]*batch_size*num_samples]
            eps = self.diffnet(x_t, beta, context)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x_t)
            first_term = (ab_prev / ab_cur)**0.5 * x_t
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x_t = first_term + second_term + third_term
        all_outputs = x_t.reshape(batch_size, num_samples, num_points, point_dim)
        return all_outputs

    def sample_at_tau_batch(self, num_samples, all_context, x_tau=None, tau=5):
        batch_size, num_samples, ph, out_dims = x_tau.shape
        all_outputs = []
        all_context = all_context.repeat(num_samples, 1)
        context = all_context
        x_T = x_tau.reshape(batch_size*num_samples, ph, out_dims)
        ts = np.linspace(self.var_sched.num_steps, 0, (self.ddim_sample_step + 1))
        simple_var = False
        eta = 0
        if simple_var:
            eta = 1
        x_t = x_T
        for i in range(self.ddim_sample_step+1-tau, self.ddim_sample_step+1):
            cur_t = int(ts[i - 1])
            prev_t = int(ts[i])
            ab_cur = self.var_sched.alpha_bars[cur_t]
            ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
            beta = self.var_sched.betas[[cur_t]*batch_size*num_samples]
            eps = self.diffnet(x_t, beta, context)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x_t)
            first_term = (ab_prev / ab_cur)**0.5 * x_t
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x_t = first_term + second_term + third_term
        all_outputs = x_t
        all_outputs = all_outputs.reshape(batch_size, num_samples, ph, out_dims)
        return all_outputs
