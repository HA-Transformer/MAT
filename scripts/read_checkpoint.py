#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Checkpoint reader.')
    parser.add_argument('path', help='Checkpoint path')
    parser.add_argument('-k', '--key', action='store_true', default=False, help='Print all keys')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    checkpoint = torch.load(args.path, map_location='cpu')
    model = checkpoint['model']

    print(checkpoint['args'])
    print(checkpoint.keys())
    print(checkpoint['extra_state'])

    if args.key:
        for key in model:
            print(key)


if __name__ == '__main__':
    main()
