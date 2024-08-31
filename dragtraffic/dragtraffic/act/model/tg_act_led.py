import copy
import logging
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR
from dragtraffic.act.utils.model_utils import act_loss
from dragtraffic.act.model.tg_act_diff import actuatordiff
from dragtraffic.act.model.tg_act_drag import actuatordrag

import torch.nn as nn

logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0

class actuatorled(pl.LightningModule):
    """ A transformer model with wider latent space """
    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_diff_is_frozen = False  
        self.num_modes = cfg['num_modes']
        self.unfreeze_at_model_epoch = cfg['unfreeze_at_model_epoch']
        self.NUM_Tau = cfg['NUM_Tau']
        init_model_path = cfg['init_model_path']
        
        if init_model_path != None:
            print('load pretrained init_model from {}...'.format(init_model_path))
            self.model_initializer = actuatordrag.load_from_checkpoint(init_model_path)
        else:
            print('Initializing model from scratch...')
            self.model_initializer = actuatordrag(cfg['init_model_cfg'])
           
        diff_model_path = cfg['diff_model_path']
        if cfg['diff_model_path'] != None:
            print('Loading pretrained diff model from {}...'.format(diff_model_path))
            self.model_diff = actuatordiff.load_from_checkpoint(diff_model_path)
            if cfg['freeze_diff']:
                print('Freezing diff model...')
                self.model_diff.freeze()
                self.model_diff_is_frozen = True
                print('Will unfreeze model at epoch: ', self.unfreeze_at_model_epoch)
                print('Be careful that the model unfreezing will consume extra GPU memory, \
                    please adjust the batch size accordingly.')
        else:
            print('Initializing diff model from scratch...')
            self.model_diff = actuatordiff(cfg['diff_model_cfg'])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.2, verbose=True)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_init = self.model_initializer.forward_led(batch)
        xyh_init = torch.cat([pred_init['pos'], pred_init['velo'], pred_init['heading'].unsqueeze(-1)], dim=-1)

        model_diff_embed = self.model_diff.encode(batch)
        pred_refine = self.model_diff.sample_at_tau_batch(self.num_modes, model_diff_embed, x_tau=xyh_init, tau=self.NUM_Tau)
        pred = {}
        pred['prob'] = pred_init['prob']
        pred['pos'] = pred_refine[..., :2].cumsum(-2)
        pred['velo'] = pred_refine[..., 2:4]
        pred['heading'] = pred_refine[..., -1].cumsum(-1)

        loss, loss_dict_act = act_loss(pred, batch, num_modes=self.num_modes)
        loss_dict.update(loss_dict_act)
    
        loss_dict = {'train/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = {}
        pred_init = self.model_initializer.forward_led(batch)
        xyh_init = torch.cat([pred_init['pos'], pred_init['velo'], pred_init['heading'].unsqueeze(-1)], dim=-1)

        model_diff_embed = self.model_diff.encode(batch)
        pred_refine = self.model_diff.sample_at_tau_batch(self.num_modes, model_diff_embed, x_tau=xyh_init, tau=self.NUM_Tau)

        pred = {}
        pred['prob'] = pred_init['prob']
        pred['pos'] = pred_refine[..., :2].cumsum(-2)
        pred['velo'] = pred_refine[..., 2:4]
        pred['heading'] = pred_refine[..., -1].cumsum(-1)

        loss, loss_dict_act = act_loss(pred, batch, num_modes=self.num_modes)
        loss_dict.update(loss_dict_act)
    
        loss_dict = {'val/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)

        return loss

    def test_step(self, batch, sample_num=None, not_from_drag=False):
        if sample_num is None:
            sample_num = self.num_modes

        pred_init = self.model_initializer.forward_led(batch, not_from_drag=not_from_drag)
        xyh_init = torch.cat([pred_init['pos'], pred_init['velo'], pred_init['heading'].unsqueeze(-1)], dim=-1)

        model_diff_embed = self.model_diff.encode(batch, not_from_drag=not_from_drag)
        pred_refine = self.model_diff.sample_at_tau_batch(sample_num, model_diff_embed, x_tau=xyh_init, tau=self.NUM_Tau)

        pred = {}
        pred['prob'] = pred_init['prob']
        pred['pos'] = pred_refine[..., :2].cumsum(-2)
        pred['velo'] = pred_refine[..., 2:4]
        pred['heading'] = pred_refine[..., -1].cumsum(-1)

        return pred

    def on_train_epoch_start(self):
        if self.current_epoch >= self.unfreeze_at_model_epoch and self.model_diff_is_frozen:
            if self.cfg['init_model_path'] != '':
                self.model_diff.unfreeze()
            self.model_diff_is_frozen = False
            print('At epoch {} Unfreezing model...'.format(self.current_epoch))
