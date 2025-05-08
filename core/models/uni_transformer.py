import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product
from torch.nn.utils.rnn import pad_sequence

class CrossAttention(nn.Module):
    """
    add cross attention
    """

    def __init__(self, input_dim, n_embd, out_dim, n_head, attn_pdrop=0., resid_pdrop=0., cross_pdrop=0., mode='cross'):
        super().__init__()
        assert n_embd % n_head == 0

        # cross attention
        self.cross_key = nn.Linear(out_dim, out_dim)
        self.cross_query = nn.Linear(out_dim, out_dim)
        self.cross_value = nn.Linear(out_dim, out_dim)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.cross_crop = nn.Dropout(cross_pdrop)
        # output projection
        self.proj = nn.Linear(out_dim, out_dim)

        num = 0

        self.n_head = n_head

        if mode == 'concat':
            self.concat_proj = nn.Sequential(
                nn.Linear(n_embd + 512, n_embd),
                nn.GELU(),
                nn.Linear(n_embd, n_embd),
                nn.LayerNorm(n_embd),
                nn.Dropout(cross_pdrop),
            )

        elif mode == 'cross':
            self.cross_attn_proj = nn.Sequential(
                nn.Linear(input_dim, n_embd),
                nn.GELU(),
                nn.Linear(n_embd, out_dim),
                nn.LayerNorm(out_dim),
                nn.Dropout(cross_pdrop),
            )
        else:
            raise ValueError('mode should be "concat" or "cross"')

    def forward(self, x, batch_x, batch_encoder, layer_past=None, encoder_embedding=None, encoder_mask=None, mode='cross'):
        unique_ids_x = batch_x.unique()
        grouped_x = [x[batch_x == uid] for uid in unique_ids_x]
        lengths_x = [seq.size(0) for seq in grouped_x]
        padded_x = pad_sequence(grouped_x, batch_first=True)
        B, T, C = padded_x.size()
        if encoder_mask is not None and encoder_mask is not None:
            unique_ids_encoder = batch_encoder.unique()
            grouped_encoder = [encoder_embedding[batch_encoder == uid] for uid in unique_ids_encoder]
            lengths_encoder = [seq.size(0) for seq in grouped_encoder]
            padded_encoder = pad_sequence(grouped_encoder, batch_first=True)

            encoder_mask = encoder_mask.to(x.device)
            grouped_encoder_mask = [encoder_mask[batch_encoder == uid] for uid in unique_ids_encoder]
            encoder_mask = pad_sequence(grouped_encoder_mask, batch_first=True)

        if encoder_embedding is not None and encoder_mask is not None:
            ## option 1: pool encoder_embedding to a single vector, then concat it to each position and use MLP to adjust dimension
            if mode == 'concat':
                # import pdb; pdb.set_trace()
                encoder_embedding = encoder_embedding.sum(axis=1)

                # assume encoder_embedding shape is (B, C), we expand it to (B, T, C)
                encoder_embedding_expanded = encoder_embedding.unsqueeze(1).expand(-1, T, -1)
                cross_att_in = torch.cat([x, encoder_embedding_expanded], dim=-1)
                # import pdb; pdb.set_trace()
                cross_att_in = self.concat_proj(cross_att_in)

                # cross attention key, query, value
                cross_k = self.cross_key(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_q = self.cross_query(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_v = self.cross_value(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

                # calculate cross attention weights
                cross_att = (cross_q @ cross_k.transpose(-2, -1)) * (1.0 / math.sqrt(cross_k.size(-1)))
                cross_att = F.softmax(cross_att, dim=-1)
                cross_att = self.attn_drop(cross_att)
                cross_y = cross_att @ cross_v
                cross_y = cross_y.transpose(1, 2).contiguous().view(B, T, C)
                # import pdb; pdb.set_trace()

            elif mode == 'cross':
                ## option 2: encoder_embedding is a sequence of vectors, then use encoder_mask to mask the padding
                # encoder_embedding shape is (B, S, C)£¬S is sequence length after padding
                # encoder_mask shape is (B, S)£¬valid position is 1£¬padding position is 0
                # import pdb; pdb.set_trace()
                B, S, _ = padded_encoder.size()
                padded_encoder = self.cross_attn_proj(padded_encoder)
                # key, query, value
                cross_k = self.cross_key(padded_encoder).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
                cross_q = self.cross_query(padded_x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_v = self.cross_value(padded_encoder).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

                # calculate cross attention weights
                cross_att = (cross_q @ cross_k.transpose(-2, -1)) * (1.0 / math.sqrt(cross_k.size(-1)))
                # import pdb; pdb.set_trace()
                # apply encoder_mask to mask the padding position
                if encoder_mask is not None:
                    # expand mask to match attention score shape (B, 1, 1, S)
                    encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
                    # set the attention score of padding position to negative infinity
                    cross_att = cross_att.masked_fill(encoder_mask == 0, -float('inf'))

                # softmax
                cross_att = F.softmax(cross_att, dim=-1)
                cross_att = self.attn_drop(cross_att)

                cross_y = cross_att @ cross_v
                cross_y = cross_y.transpose(1, 2).contiguous().view(B, T, C)

            else:
                raise ValueError('mode should be "concat" or "cross"')

        # output projection
        y = self.resid_drop(self.proj(cross_y))
        y_unpadded = torch.cat([y[i, :l] for i, l in enumerate(lengths_x)], dim=0)
        return y_unpadded


class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            h_cross_atten_layer = CrossAttention(input_dim=512, n_embd=self.hidden_dim, out_dim=self.hidden_dim, n_head=max(int(self.n_heads/2), 1))  # input_dim is the lig embedding dim
            x_cross_atten_layer = CrossAttention(input_dim=512, n_embd=128, out_dim=3, n_head=1)
            base_block.append(nn.Sequential(layer, h_cross_atten_layer, x_cross_atten_layer))
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.knn, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch_lig, batch_all, lig_embedding, embedding_mask, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch_all)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h_g = layer[1](x=h, batch_x=batch_all, batch_encoder=batch_lig, encoder_embedding=lig_embedding, encoder_mask=embedding_mask)
                h = h + h_g
                x_g = layer[2](x=x, batch_x=batch_all, batch_encoder=batch_lig, encoder_embedding=lig_embedding, encoder_mask=embedding_mask)
                x = x + x_g * mask_ligand[:, None]
                h, x = layer[0](h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
