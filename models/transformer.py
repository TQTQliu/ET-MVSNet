import torch
import copy
import torch.nn.functional as F
from torch import nn

class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1,
                 activation="relu"):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, ref, src, mask_ref, mask_src, pos_ref, pos_src):
        '''
        src: bs x N x D
        pos_src : bs x N X D
        '''
        src = src.permute(1, 0, 2)
        ref = ref.permute(1, 0, 2)

        pos_src = pos_src.permute(1, 0, 2)
        pos_ref = pos_ref.permute(1, 0, 2)

        memory = self.encoder(ref,src,mask_ref,mask_src,pos_ref,pos_src)
        return memory

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, ref, src,
                mask_ref, mask_src,
                pos_ref, pos_src):

        for layer in self.layers:
            ref,src = layer(ref,src,mask_ref,mask_src,pos_ref,pos_src)

        return ref, src
        
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.iea = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cea = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                ref, src,
                mask_ref,mask_src,
                pos_ref,pos_src):
        # src self-attention
        q = k = self.with_pos_embed(src, pos_src)
        src_iea = self.iea(q, k, value=src, attn_mask=None,
                              key_padding_mask=mask_src)[0]
        src = src + self.dropout(src_iea)
        src = self.norm1(src)


        # cross-attention(q:src,k\v:ref)
        q = self.with_pos_embed(src, pos_src)
        k = self.with_pos_embed(ref, pos_ref)
        src_cea = self.cea(q, k, value=ref, attn_mask=None,
                              key_padding_mask=mask_ref)[0]

        src = src + self.dropout(src_cea)
        src = self.norm2(src)

        src_ = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src_)
        src = self.norm3(src)

        return ref,src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])