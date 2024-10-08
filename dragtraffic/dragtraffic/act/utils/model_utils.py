import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from sklearn.cluster import KMeans
import psutil
import ray
from scipy.spatial.distance import cdist


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask, tgt_mask, query_pos=None):
        """
        Take in and process masked src and target sequences.
        """
        output = self.encode(src, src_mask)
        return self.decode(output, src_mask, tgt, tgt_mask, query_pos)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, query_pos=None):
        return self.decoder(tgt, memory, src_mask, tgt_mask, query_pos)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PE():
    """
    Implement the PE function.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, n, return_intermediate=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)
        self.return_intermediate = return_intermediate

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):

        intermediate = []

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(x))

        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # TODO How to fusion the feature
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):
        """
        Follow Figure 1 (right) for connections.
        """
        m = memory
        q = k = self.with_pos_embed(x, query_pos)
        x = self.sublayer[0](x, lambda x: self.self_attn(q, k, x, tgt_mask))
        x = self.with_pos_embed(x, query_pos)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if len(query.shape) > 3:
            batch_dim = len(query.shape) - 2
            batch = query.shape[:batch_dim]
            mask_dim = batch_dim
        else:
            batch = (query.shape[0], )
            mask_dim = 1
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(dim=mask_dim)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(*batch, -1, self.h, self.d_k).transpose(-3, -2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-3, -2).contiguous().view(*batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class MCG_block(nn.Module):
    def __init__(self, hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, inp, context, mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp = inp * context
        inp = inp.masked_fill(mask == 0, torch.tensor(-1e9))
        context = torch.max(inp, dim=1)[0]
        return inp, context


class CG_stacked(nn.Module):
    def __init__(self, stack_num, hidden_dim):
        super(CG_stacked, self).__init__()
        self.CGs = nn.ModuleList()
        self.stack_num = stack_num
        for i in range(stack_num):
            self.CGs.append(MCG_block(hidden_dim))

    def forward(self, inp, context, mask):

        inp_, context_ = self.CGs[0](inp, context, mask)
        for i in range(1, self.stack_num):
            inp, context = self.CGs[i](inp_, context_, mask)
            inp_ = (inp_ * i + inp) / (i + 1)
            context_ = (context_ * i + context) / (i + 1)
        return inp_, context_


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)

    # Q,K,V: [bs,h,num,dim]
    # scores: [bs,h,num1,num2]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask: [bs,1,1,num2] => dimension expansion

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_unit), nn.LayerNorm(hidden_unit), nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLP_FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LocalSubGraphLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        """Local subgraph layer

        :param dim_in: input feat size
        :type dim_in: int
        :param dim_out: output feat size
        :type dim_out: int
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = MLP(dim_in, dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the model

        :param x: input tensor
        :tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x
        :tensor invalid_mask (B,N,P)
        :return: output tensor (B,N,P,dim_out)
        :rtype: torch.Tensor
        """
        # x input -> polys * num_vectors * embedded_vector_length
        _, num_vectors, _ = x.shape
        # x mlp -> polys * num_vectors * dim_in
        x = self.mlp(x)
        # compute the masked max for each feature in the sequence

        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float("-inf"))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        # repeat it along the sequence length
        x_agg = x_agg.repeat(1, num_vectors, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)  # remap to a possibly different feature length
        return x


class LocalSubGraph(nn.Module):
    def __init__(self, num_layers: int, dim_in: int) -> None:
        """PointNet-like local subgraph - implemented as a collection of local graph layers

        :param num_layers: number of LocalSubGraphLayer
        :type num_layers: int
        :param dim_in: input, hidden, output dim for features
        :type dim_in: int
        """
        super(LocalSubGraph, self).__init__()
        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        for _ in range(num_layers):
            self.layers.append(LocalSubGraphLayer(dim_in, dim_in))

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the module:
        - Add positional encoding
        - Forward to layers
        - Aggregates using max
        (calculates a feature descriptor per element - reduces over points)

        :param x: input tensor (B,N,P,dim_in)
        :type x: torch.Tensor
        :param invalid_mask: invalid mask for x (B,N,P)
        :type invalid_mask: torch.Tensor
        :param pos_enc: positional_encoding for x
        :type pos_enc: torch.Tensor
        :return: output tensor (B,N,P,dim_in)
        :rtype: torch.Tensor
        """
        batch_size, polys_num, seq_len, vector_size = x.shape
        invalid_mask = ~invalid_mask
        # exclude completely invalid sequences from local subgraph to avoid NaN in weights
        x_flat = x.view(-1, seq_len, vector_size)
        invalid_mask_flat = invalid_mask.view(-1, seq_len)
        # (batch_size x (1 + M),)
        valid_polys = ~invalid_mask.all(-1).flatten()
        # valid_seq x seq_len x vector_size
        x_to_process = x_flat[valid_polys]
        mask_to_process = invalid_mask_flat[valid_polys]
        for layer in self.layers:
            x_to_process = layer(x_to_process, mask_to_process)

        # aggregate sequence features
        x_to_process = x_to_process.masked_fill(mask_to_process[..., None] > 0, float("-inf"))
        # valid_seq x vector_size
        x_to_process = torch.max(x_to_process, dim=1).values

        # restore back the batch
        x = torch.zeros_like(x_flat[:, 0])
        x[valid_polys] = x_to_process
        x = x.view(batch_size, polys_num, self.dim_in)
        return x


class MLP_3(nn.Module):
    def __init__(self, dims):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.LayerNorm(dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]), nn.ReLU(), nn.Linear(dims[2], dims[3])
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f'lmlp_{i}', MLP(in_channels, in_channels))
            # in_channels = hidden_unit * 2

    def forward(self, lane):
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)

        x_max = torch.max(x, -2)[0]
        return x_max


def split_dim(x: torch.Tensor, split_shape: tuple, dim: int):
    if dim < 0:
        dim = len(x.shape) + dim
    return x.reshape(*x.shape[:dim], *split_shape, *x.shape[dim + 1:])

class SqrtLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(torch.mean((y_pred - y_true)**2))
        return loss

def act_loss(pred, gt, pose_norm=50, speed_norm=30, num_modes=6):

    MSE = torch.nn.MSELoss(reduction='none')
    L1 = torch.nn.L1Loss(reduction='none')
    CLS = torch.nn.CrossEntropyLoss()
    loss_sum = 0
    cls_loss = 0
    velo_loss = 0
    heading_loss = 0

    prob_pred = pred['prob'] if 'prob' in pred.keys() else None
    velo_pred = pred['velo'] if 'velo' in pred.keys() else None
    pos_pred = pred['pos'] # torch.Size([96, 6, 89, 2])
    heading_pred = pred['heading'] if 'heading' in pred.keys() else None

    # ego_mask_hp = torch.ones_like(heading_pred)
    if 'ego_mask' in gt.keys():
        ego_mask_hp = gt['ego_mask'][:, 1:].unsqueeze(1).expand(-1, num_modes, -1)

    # if 'velo' in pred.keys():
    ego_mask_vp_pp = ego_mask_hp.unsqueeze(-1).expand(-1, -1, -1, pos_pred.size(-1))
    # bf_end_point_index = gt['bf_end_point_index'] # unsqueeze(1).repeat(1, 6, 1)

    pos_gt = gt['gt_pos'][:, 1:].unsqueeze(1).repeat(1, num_modes, 1, 1)
    velo_gt = gt['gt_vel'][:, 1:].unsqueeze(1).repeat(1, num_modes, 1, 1)
    heading_gt = gt['gt_heading'][:, 1:].unsqueeze(1).repeat(1, num_modes, 1)
    
    # bf_end_point_index = bf_end_point_index.unsqueeze(1).unsqueeze(1).repeat(1, 6, 1, 2)
    pred_end = pos_pred[:, :, -1]
    gt_end = pos_gt[:, :, -1]
    dist = MSE(pred_end, gt_end).mean(-1)
    min_index = torch.argmin(dist, dim=-1)

    if 'prob' in pred.keys():
        cls_loss = CLS(prob_pred, min_index)
        loss_sum += cls_loss

    ade = MSE(pos_gt*ego_mask_vp_pp, pos_pred*ego_mask_vp_pp).sum(-1)
    pos_loss = torch.gather(ade.sum(-1), dim=1, index=min_index.unsqueeze(-1))
    fde = torch.gather(ade[..., -1], dim=1, index=min_index.unsqueeze(-1))

    pos_loss = (pos_loss.sum() / (ego_mask_vp_pp[:,0,:,0].sum().item() + 1e-6))*pose_norm
    fde = (fde.sum() / (ego_mask_vp_pp[:,0,-1,0].sum() + 1e-6))*pose_norm

    loss_sum += pos_loss 

    if 'velo' in pred.keys():
        velo_loss = MSE(velo_gt*ego_mask_vp_pp, velo_pred*ego_mask_vp_pp).sum(-1)
        velo_loss = torch.gather(velo_loss.sum(-1), dim=1, index=min_index.unsqueeze(-1))
        velo_loss = (velo_loss.sum() / (ego_mask_vp_pp[:,0,:,0].sum() + 1e-6))*speed_norm
        loss_sum += velo_loss

    if 'heading' in pred.keys():
        heading_loss = L1(heading_gt*ego_mask_hp, heading_pred*ego_mask_hp)
        heading_loss = torch.gather(heading_loss.sum(-1), dim=1, index=min_index.unsqueeze(-1))
        heading_loss = heading_loss.sum() / (ego_mask_hp[:,0].sum() + 1e-6)
        loss_sum += heading_loss
    

    loss_dict = {}
    loss_dict['cls_loss'] = cls_loss
    loss_dict['velo_loss'] = velo_loss
    loss_dict['heading_loss'] = heading_loss
    loss_dict['fde'] = fde
    loss_dict['pos_loss'] = pos_loss

    loss_dict['ade'] = (torch.sqrt(pos_loss*pose_norm)).item()
    loss_dict['fde'] = (torch.sqrt(fde*pose_norm)).item()

    # print(loss_sum, loss_dict)
    return loss_sum, loss_dict


# def act_loss(pred, gt, pose_norm=50, speed_norm=30, num_modes=6):
#     MSE = torch.nn.MSELoss(reduction='none')
#     L1 = torch.nn.L1Loss(reduction='none')
#     CLS = torch.nn.CrossEntropyLoss()

#     prob_pred = pred['prob']
#     velo_pred = pred['velo']
#     pos_pred = pred['pos']
#     heading_pred = pred['heading']

#     pos_gt = gt['gt_pos'][:, 1:, :].unsqueeze(1).repeat(1, num_modes, 1, 1)
#     velo_gt = gt['gt_vel'][:, 1:, :].unsqueeze(1).repeat(1, num_modes, 1, 1)
#     heading_gt = gt['gt_heading'][:, 1:].unsqueeze(1).repeat(1, num_modes, 1)

#     pred_end = pos_pred[:, :, -1]
#     gt_end = pos_gt[:, :, -1]
#     dist = MSE(pred_end, gt_end).mean(-1)
#     min_index = torch.argmin(dist, dim=-1)

#     cls_loss = CLS(prob_pred, min_index)

#     pos_loss = MSE(pos_gt, pos_pred).mean(-1).mean(-1)
#     fde = MSE(pos_gt, pos_pred).mean(-1)[..., -1]
#     pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()*pose_norm**2
#     fde = torch.gather(fde, dim=1, index=min_index.unsqueeze(-1)).mean()*pose_norm**2

#     velo_loss = MSE(velo_gt, velo_pred).mean(-1).mean(-1)
#     velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()*speed_norm**2

#     heading_loss = L1(heading_gt, heading_pred).mean(-1)
#     heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

#     loss_sum = pos_loss + velo_loss + heading_loss + cls_loss

#     loss_dict = {}
#     loss_dict['cls_loss'] = cls_loss
#     loss_dict['velo_loss'] = velo_loss
#     loss_dict['heading_loss'] = heading_loss
#     loss_dict['fde'] = fde
#     loss_dict['pos_loss'] = pos_loss

#     loss_dict['ade'] = (torch.sqrt(pos_loss)).item()
#     loss_dict['fde'] = (torch.sqrt(fde)).item()

#     return loss_sum, loss_dict



# def act_loss(pred, gt, pose_norm=50, speed_norm=30, num_modes=6):
#     MSE = torch.nn.MSELoss(reduction='none')
#     L1 = torch.nn.L1Loss(reduction='none')
#     CLS = torch.nn.CrossEntropyLoss()

#     prob_pred = pred['prob'] if 'prob' in pred.keys() else None
#     velo_pred = pred['velo']
#     pos_pred = pred['pos']
#     heading_pred = pred['heading']

#     cls_loss = 0
#     velo_loss = 0
#     heading_loss = 0
#     pos_loss = 0

#     pos_gt = gt['gt_pos'][:, 1:, :].unsqueeze(1).repeat(1, num_modes, 1, 1)
#     velo_gt = gt['gt_vel'][:, 1:, :].unsqueeze(1).repeat(1, num_modes, 1, 1)
#     heading_gt = gt['gt_heading'][:, 1:].unsqueeze(1).repeat(1, num_modes, 1)

#     pred_end = pos_pred[:, :, -1]
#     gt_end = pos_gt[:, :, -1]
#     dist = MSE(pred_end, gt_end).mean(-1)
#     min_index = torch.argmin(dist, dim=-1)

#     cls_loss = CLS(prob_pred, min_index) if prob_pred is not None else 0

#     pos_loss = MSE(pos_gt, pos_pred).mean(-1).mean(-1) * pose_norm
#     fde = MSE(pos_gt, pos_pred).mean(-1)[..., -1] * pose_norm
#     pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
#     fde = torch.gather(fde, dim=1, index=min_index.unsqueeze(-1)).mean()

#     velo_loss = MSE(velo_gt, velo_pred).mean(-1).mean(-1) * speed_norm
#     velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

#     heading_loss = L1(heading_gt, heading_pred).mean(-1)
#     heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

#     loss_sum = pos_loss + velo_loss + heading_loss + cls_loss

#     loss_dict = {}
#     loss_dict['cls_loss'] = cls_loss
#     loss_dict['velo_loss'] = velo_loss
#     loss_dict['heading_loss'] = heading_loss
#     loss_dict['fde'] = fde
#     loss_dict['pos_loss'] = pos_loss

#     loss_dict['ade'] = (torch.sqrt(pos_loss * pose_norm)).item()
#     loss_dict['fde'] = (torch.sqrt(fde * pose_norm)).item()

#     return loss_sum, loss_dict








# # Initialize device:
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Initialize ray:
# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus, log_to_driver=False)


def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


@ray.remote
def cluster_and_rank(k: int, data: np.ndarray):
    """
    Combines the clustering and ranking steps so that ray.remote gets called just once
    """

    def cluster(n_clusters: int, x: np.ndarray):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(n_clusters=n_clusters, n_init=1, max_iter=100, init='random').fit(x)
        return clustering_op.labels_, clustering_op.cluster_centers_

    def rank_clusters(cluster_counts, cluster_centers):
        """
        Rank the K clustered trajectories using Ward's criterion. Start with K cluster centers and cluster counts.
        Find the two clusters to merge based on Ward's criterion. Smaller of the two will get assigned rank K.
        Merge the two clusters. Repeat process to assign ranks K-1, K-2, ..., 2.
        """

        num_clusters = len(cluster_counts)
        cluster_ids = np.arange(num_clusters)
        ranks = np.ones(num_clusters)

        for i in range(num_clusters, 0, -1):
            # Compute Ward distances:
            centroid_dists = cdist(cluster_centers, cluster_centers)
            n1 = cluster_counts.reshape(1, -1).repeat(len(cluster_counts), axis=0)
            n2 = n1.transpose()
            wts = n1 * n2 / (n1 + n2)
            dists = wts * centroid_dists + np.diag(np.inf * np.ones(len(cluster_counts)))

            # Get clusters with min Ward distance and select cluster with fewer counts
            c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
            c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
            c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1

            # Assign rank i to selected cluster
            ranks[cluster_ids[c]] = i

            # Merge clusters and update identity of merged cluster
            cluster_centers[c_] = (cluster_counts[c_] * cluster_centers[c_] + cluster_counts[c] * cluster_centers[c]) /\
                                  (cluster_counts[c_] + cluster_counts[c])
            cluster_counts[c_] += cluster_counts[c]

            # Discard merged cluster
            cluster_ids = np.delete(cluster_ids, c)
            cluster_centers = np.delete(cluster_centers, c, axis=0)
            cluster_counts = np.delete(cluster_counts, c)

        return ranks

    cluster_lbls, cluster_ctrs = cluster(k, data)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
    return {'lbls': cluster_lbls, 'ranks': cluster_ranks, 'counts': cluster_cnts}


def cluster_traj(k: int, traj: torch.Tensor, device):
    """
    clusters sampled trajectories to output K modes.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]

    # Down-sample traj along time dimension for faster clustering
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    # Cluster and rank
    cluster_ops = ray.get([cluster_and_rank.remote(k, data_slice) for data_slice in data])
    cluster_lbls = [cluster_op['lbls'] for cluster_op in cluster_ops]
    cluster_counts = [cluster_op['counts'] for cluster_op in cluster_ops]
    cluster_ranks = [cluster_op['ranks'] for cluster_op in cluster_ops]

    # Compute mean (clustered) traj and scores
    lbls = torch.as_tensor(cluster_lbls, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2).long()
    traj_summed = torch.zeros(batch_size, k, traj_len, 2, device=device).scatter_add(1, lbls, traj)
    cnt_tensor = torch.as_tensor(cluster_counts, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)
    traj_clustered = traj_summed / cnt_tensor
    scores = 1 / torch.as_tensor(cluster_ranks, device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores
