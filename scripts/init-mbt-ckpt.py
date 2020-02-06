#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Initialize the Multi-branch Transformer checkpoints with corresponding single-branch Transformer checkpoints.
Used for warm start MBT training.
"""

import argparse
import collections
import re

import torch

from fairseq.checkpoint_utils import torch_persistent_save

_ATTN_BRANCHES_KEY_FULL = re.compile(r'.*attn_branches\.\d+\..*')
_ATTN_BRANCHES_KEY = re.compile(r'attn_branches\.\d+\.')
_PFFN_BRANCHES_KEY_FULL = re.compile(r'.*fc[12]_branches\.\d+\..*')
_PFFN_BRANCHES_KEY = re.compile(r'fc([12])_branches\.\d+\.')


def main():
    parser = argparse.ArgumentParser('Initialize the MBT checkpoint with corresponding single-branch Transformer checkpoints.')
    parser.add_argument('path', help='Original checkpoint path.')
    parser.add_argument('output_path', help='The output checkpoint path.')
    parser.add_argument('-N', type=int, help='New N value')
    parser.add_argument('--NF', type=int, help='New NF value')
    parser.add_argument('--ro', '--reset-optimizer', dest='reset_optimizer',
                        action='store_true', default=False, help='Reset optimizer states.')

    args = parser.parse_args()

    checkpoint = torch.load(args.path, map_location='cpu')
    model = checkpoint['model']
    pt_args = checkpoint['args']

    print('| Init MBT checkpoint: convert {oldN}-{oldNF}-{dH}-{dF} to {N}-{NF}-{dH}-{dF}'.format(
        oldN=pt_args.encoder_branches, oldNF=pt_args.encoder_pffn_branches,
        dH=pt_args.encoder_embed_dim, dF=pt_args.encoder_ffn_embed_dim,
        N=args.N, NF=args.NF))
    
    new_model = collections.OrderedDict()
    
    for key, param in model.items():
        if _ATTN_BRANCHES_KEY_FULL.fullmatch(key) is not None:
            for i in range(args.N):
                new_key = _ATTN_BRANCHES_KEY.sub(r'attn_branches.{}.'.format(i), key, count=1)
                new_model[new_key] = param.clone()
        elif _PFFN_BRANCHES_KEY_FULL.fullmatch(key) is not None:
            for i in range(args.NF):
                new_key = _PFFN_BRANCHES_KEY.sub(r'fc\1_branches.{}.'.format(i), key, count=1)
                new_model[new_key] = param.clone()
        else:
            new_model[key] = param
    
    checkpoint['model'] = new_model
    pt_args.encoder_branches = args.N
    pt_args.decoder_branches = args.N
    pt_args.encoder_pffn_branches = args.NF
    pt_args.decoder_pffn_branches = args.NF

    if args.reset_optimizer:
        checkpoint['extra_state'] = None
        checkpoint['optimizer_history'] = []
        checkpoint['last_optimizer_state'] = None
        checkpoint['warm_init'] = True

    torch_persistent_save(checkpoint, args.output_path)
    print('| Init MBT checkpoint: saved to {!r}.'.format(args.output_path))


if __name__ == "__main__":
    main()
