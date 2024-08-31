import copy
import logging
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR
from dragtraffic.act.utils.model_utils import MLP_3, CG_stacked, act_loss

import torch.nn as nn

logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0

class actuatordrag(pl.LightningModule):
    """ A transformer model with wider latent space """
    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        # input embedding stem
        hidden_dim = 1024
        self.num_modes = self.cfg['num_modes'] if 'num_modes' in self.cfg else 6
        self.feature_dim = 2
        self.CG_agent = CG_stacked(5, hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        hidden_dim_condition_context = 0
        if self.cfg['model'] == 'drag' and self.cfg['condition_context']:
            hidden_dim_condition_context = hidden_dim
            self.condition_context_encode = MLP_3([8, int(hidden_dim_condition_context/4), int(hidden_dim_condition_context/2), hidden_dim_condition_context])
            self.feature_dim = 2

        self.CG_all = CG_stacked(5, hidden_dim * self.feature_dim + hidden_dim_condition_context)
        self.agent_encode = MLP_3([8, 256, 512, hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)
        self.anchor_embedding = nn.Embedding(self.num_modes, hidden_dim * self.feature_dim + hidden_dim_condition_context)

        self.pred_len = 89
        self.velo_head = MLP_3([hidden_dim * self.feature_dim + hidden_dim_condition_context, hidden_dim, 256, self.pred_len * 2])
        self.pos_head = MLP_3([hidden_dim * self.feature_dim + hidden_dim_condition_context, hidden_dim, 256, self.pred_len * 2])
        self.angle_head = MLP_3([hidden_dim * self.feature_dim + hidden_dim_condition_context, hidden_dim, 256, self.pred_len])

        self.prob_head = MLP_3([hidden_dim * self.feature_dim + hidden_dim_condition_context, hidden_dim, 256, 1])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = {}
        loss, loss_dict_act = act_loss(pred, batch, num_modes=self.num_modes)
        loss_dict.update(loss_dict_act)

        loss_dict = {'train/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_dict = {}
        loss, loss_dict_act = act_loss(pred, batch, num_modes=self.num_modes)
        loss_dict.update(loss_dict_act)
        loss_dict = {'val/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.2, verbose=True)
        return [optimizer], [scheduler]

    def forward(self, data):
        agent = data['agent']
        agent_mask = data['agent_mask']

        if self.cfg['condition_context']:
            gt_pos = data['gt_pos'][:,-1]
            gt_vel = data['gt_vel'][:,-1]
            gt_heading = data['gt_heading'][:,-1].unsqueeze(-1)
            type_l_w = data['agent'][:, 0, 5:8]
            condition_context = torch.cat([gt_pos, gt_vel, gt_heading, type_l_w], dim=-1)

        all_vec = torch.cat([data['center'], data['cross'], data['bound']], dim=-2)
        line_mask = torch.cat([data['center_mask'], data['cross_mask'], data['bound_mask']], dim=1)

        agent_enc = self.agent_encode(agent)
        b, a, d = agent_enc.shape
        device = agent_enc.device
        context_agent = torch.ones([b, d]).to(device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        polyline = all_vec[..., :4]
        polyline_type = all_vec[..., 4].to(int)
        polyline_traf = all_vec[..., 5].to(int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)

        if self.cfg['model'] == 'drag' and self.cfg['condition_context']:
            condition_context = self.condition_context_encode(condition_context)
            all_context = torch.cat([condition_context, context_agent, context_line], dim=-1)
        else:
            all_context = torch.cat([context_agent, context_line], dim=-1)

        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_all(anchors, all_context, mask)

        prob_pred = (self.prob_head(pred_embed)).squeeze(-1)
        velo_pred = self.velo_head(pred_embed).view(b, self.num_modes, self.pred_len, 2)
        pos_pred = self.pos_head(pred_embed).view(b, self.num_modes, self.pred_len, 2).cumsum(-2)
        heading_pred = self.angle_head(pred_embed).view(b, self.num_modes, self.pred_len).cumsum(-1)

        pred = {}
        pred['prob'] = prob_pred
        pred['velo'] = velo_pred
        pred['pos'] = pos_pred
        pred['heading'] = heading_pred

        return pred

    def forward_led(self, data, not_from_drag=False):
        agent = data['agent']
        agent_mask = data['agent_mask']

        if self.cfg['condition_context']:
            goal_condition_indices = (data['gt_pos'].shape[1] -1) * torch.ones(data['gt_pos'].shape[0], dtype=torch.long).to(data['gt_pos'].device)
            if 'goal_condition_indices' in data:
                goal_condition_indices = data['goal_condition_indices'].long()
            gt_pos = torch.gather(data['gt_pos'], 1, goal_condition_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            gt_vel = torch.gather(data['gt_vel'], 1, goal_condition_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            gt_heading = torch.gather(data['gt_heading'], 1, goal_condition_indices.view(-1, 1))

            type_l_w = data['agent'][:, 0, 5:8]
            condition_context = torch.cat([gt_pos, gt_vel, gt_heading, type_l_w], dim=-1)

        all_vec = torch.cat([data['center'], data['cross'], data['bound']], dim=-2)
        line_mask = torch.cat([data['center_mask'], data['cross_mask'], data['bound_mask']], dim=1)

        agent_enc = self.agent_encode(agent)
        b, a, d = agent_enc.shape
        device = agent_enc.device
        context_agent = torch.ones([b, d]).to(device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        polyline = all_vec[..., :4]
        polyline_type = all_vec[..., 4].to(int)
        polyline_traf = all_vec[..., 5].to(int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)

        if self.cfg['model'] == 'drag' and self.cfg['condition_context']:
            condition_context = self.condition_context_encode(condition_context)
            all_context = torch.cat([condition_context, context_agent, context_line], dim=-1)
        else:
            all_context = torch.cat([context_agent, context_line], dim=-1)

        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_all(anchors, all_context, mask)

        prob_pred = (self.prob_head(pred_embed)).squeeze(-1)
        velo_pred = self.velo_head(pred_embed).view(b, self.num_modes, self.pred_len, 2)
        pos_pred = self.pos_head(pred_embed).view(b, self.num_modes, self.pred_len, 2)
        heading_pred = self.angle_head(pred_embed).view(b, self.num_modes, self.pred_len)

        pred = {}
        pred['prob'] = prob_pred
        pred['velo'] = velo_pred
        pred['pos'] = pos_pred
        pred['heading'] = heading_pred

        return pred
