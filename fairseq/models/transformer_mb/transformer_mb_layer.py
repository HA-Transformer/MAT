"""Multi-branch Transformer layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention


def merge_branches(branch_outputs, branch_dropout, training):
    """Merge branches.
    
    branch_outputs: list of Tensor or tuple of Tensors
    """
    if not isinstance(branch_outputs[0], (tuple, list)):
        branch_outputs = [(t,) for t in branch_outputs]
    branch_output_lists = tuple(zip(*branch_outputs))

    N = len(branch_outputs)
    branch_selection = branch_outputs[0][0].new(N).fill_(1.0 / N)
    branch_selection_d = F.dropout(branch_selection, p=branch_dropout, training=training)

    merged_branch_outputs = []
    for branch_output_list_i in branch_output_lists:
        if branch_output_list_i[0] is None:
            merged_branch_outputs.append(None)
        else:
            branch_output_i = torch.stack(branch_output_list_i, dim=0)
            branch_selection_d_expanded = branch_selection_d[(slice(None),) + tuple(None for _ in range(branch_output_i.ndimension() - 1))]
            merged_branch_outputs.append(torch.mean(branch_selection_d_expanded * branch_output_i, dim=0))

    if len(merged_branch_outputs) == 1:
        return merged_branch_outputs[0]
    else:
        return tuple(merged_branch_outputs)


class TransformerMBEncoderLayer(nn.Module):
    """Multi-branch Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.num_branches = args.encoder_branches
        self.num_pffn_branches = args.encoder_pffn_branches
        self.join_pffn = args.join_pffn
        self.branch_dropout = args.branch_dropout
        self.pffn_branch_dropout = args.pffn_branch_dropout
        self.enable_head_dropout = args.enable_head_dropout
        self.self_attn_branches = nn.ModuleList([
            MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True,
                head_dropout=self.branch_dropout if self.enable_head_dropout else None,
            )
            for _ in range(self.num_branches)
        ])
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1_branches = nn.ModuleList([
            Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            for _ in range(self.num_pffn_branches)
        ])
        self.fc2_branches = nn.ModuleList([
            Linear(args.encoder_ffn_embed_dim, self.embed_dim)
            for _ in range(self.num_pffn_branches)
        ])
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def self_attn(self, query, key, value, **kwargs):
        self_attn_outputs = [m(query, key, value, **kwargs) for m in self.self_attn_branches]
        return merge_branches(self_attn_outputs, self.branch_dropout, not self.enable_head_dropout and self.training)
    
    def fc1(self, x):
        fc1_outputs = [m(x) for m in self.fc1_branches]
        return merge_branches(fc1_outputs, self.pffn_branch_dropout, self.training)
    
    def fc2(self, x):
        fc2_outputs = [m(x) for m in self.fc2_branches]
        return merge_branches(fc2_outputs, self.pffn_branch_dropout, self.training)
    
    def fc1fc2(self, x):
        outputs = []
        for fc1, fc2 in zip(self.fc1_branches, self.fc2_branches):
            o = self.activation_fn(fc1(x))
            o = F.dropout(o, p=self.activation_dropout, training=self.training)
            o = fc2(o)
            outputs.append(o)
        return merge_branches(outputs, self.pffn_branch_dropout, self.training)

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        if self.join_pffn:
            x = self.fc1fc2(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerMBDecoderLayer(nn.Module):
    """Multi-branch Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.num_branches = args.decoder_branches
        self.num_pffn_branches = args.decoder_pffn_branches
        self.branch_dropout = args.branch_dropout
        self.pffn_branch_dropout = args.pffn_branch_dropout
        self.enable_head_dropout = args.enable_head_dropout
        self.join_pffn = args.join_pffn
        self.self_attn_branches = nn.ModuleList([
            MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                head_dropout=self.branch_dropout if self.enable_head_dropout else None,
            )
            for _ in range(self.num_branches)
        ])
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn_branches = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn_branches = nn.ModuleList([
                MultiheadAttention(
                    self.embed_dim,
                    args.decoder_attention_heads,
                    kdim=getattr(args, 'encoder_embed_dim', None),
                    vdim=getattr(args, 'encoder_embed_dim', None),
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                    head_dropout=self.branch_dropout if self.enable_head_dropout else None,
                )
                for _ in range(self.num_branches)
            ])
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1_branches = nn.ModuleList([
            Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            for _ in range(self.num_pffn_branches)
        ])
        self.fc2_branches = nn.ModuleList([
            Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            for _ in range(self.num_pffn_branches)
        ])

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
    
    def self_attn(self, query, key, value, **kwargs):
        self_attn_outputs = [m(query, key, value, **kwargs) for m in self.self_attn_branches]
        return merge_branches(self_attn_outputs, self.branch_dropout, not self.enable_head_dropout and self.training)
    
    def encoder_attn(self, query, key, value, **kwargs):
        encoder_attn_outputs = [m(query, key, value, **kwargs) for m in self.encoder_attn_branches]
        return merge_branches(encoder_attn_outputs, self.branch_dropout, not self.enable_head_dropout and self.training)
    
    def fc1(self, x):
        fc1_outputs = [m(x) for m in self.fc1_branches]
        return merge_branches(fc1_outputs, self.pffn_branch_dropout, self.training)
    
    def fc2(self, x):
        fc2_outputs = [m(x) for m in self.fc2_branches]
        return merge_branches(fc2_outputs, self.pffn_branch_dropout, self.training)
    
    def fc1fc2(self, x):
        outputs = []
        for fc1, fc2 in zip(self.fc1_branches, self.fc2_branches):
            o = self.activation_fn(fc1(x))
            o = F.dropout(o, p=self.activation_dropout, training=self.training)
            o = fc2(o)
            outputs.append(o)
        return merge_branches(outputs, self.pffn_branch_dropout, self.training)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            # Set input buffer for all branches.
            for self_attn in self.self_attn_branches:
                self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                # Set input buffer for all branches.
                for encoder_attn in self.encoder_attn_branches:
                    encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        if self.join_pffn:
            x = self.fc1fc2(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            # Get input buffer for all branches.
            saved_states = [self_attn._get_input_buffer(incremental_state) for self_attn in self.self_attn_branches]
            self_attn_key = torch.mean(torch.stack([saved_state['prev_key'] for saved_state in saved_states]), dim=0)
            self_attn_value = torch.mean(torch.stack([saved_state['prev_value'] for saved_state in saved_states]), dim=0)
            self_attn_state = self_attn_key, self_attn_value
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
