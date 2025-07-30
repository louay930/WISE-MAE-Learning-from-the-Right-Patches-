#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

# Increase PNG text chunk size
from PIL import Image, PngImagePlugin, Image
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

# Set NCCL-related environment variables for more robust behavior
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Ensure SummaryWriter is available
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_gcmae

from engine_pretrain import train_one_epoch
from lib.NCEAverage import NCEAverage
from lib.NCECriterion import NCECriterion

def get_args_parser():
    parser = argparse.ArgumentParser('GCMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    # Model parameters
    parser.add_argument('--model', default='gcmae_vit_base_patch16', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image input size')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio')
    parser.add_argument('--norm_pix_loss', action='store_false',
                        help='Use normalized pixels for loss')
    parser.set_defaults(norm_pix_loss=False)
    # NCE parameters
    parser.add_argument('--low_dim', default=768, type=int, help='Feature dimension')
    parser.add_argument('--nce_k', default=8192, type=int, help='Negative samples')
    parser.add_argument('--nce_t', default=0.07, type=float, help='Temperature')
    parser.add_argument('--nce_m', default=0.5, type=float, help='Momentum')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--lr', type=float, default=None,
                        help='Absolute learning rate')
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=40,
                        help='Warmup epochs')
    # Dataset parameters
    parser.add_argument('--data_path', default=' ', type=str,
                        help='Dataset path')
    parser.add_argument('--data_val_path', default=' ', type=str,
                        help='Validation dataset path')
    parser.add_argument('--output_dir', default=' ',
                        help='Output directory')
    parser.add_argument('--log_dir', default=' ',
                        help='Log directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin memory for efficient transfer')
    parser.set_defaults(pin_mem=True)
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank (set automatically by torchrun)')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='Fallback GPU id if local_rank is not provided')
    return parser

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            img = self.loader(path)
        except Exception as e:
            print(f"Skipping corrupted image: {path} | Error: {e}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index

def main(args):
    # --- Bind each process to its GPU BEFORE distributed initialization ---
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.local_rank < 0:
        raise RuntimeError("Invalid local_rank detected. Check your distributed setup.")
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    
    # Initialize distributed training (this sets args.distributed, args.rank, etc.)
    misc.init_distributed_mode(args)
    # Re-assert device in case initialization altered it
    torch.cuda.set_device(device)
    print(f"Initialized process {args.rank} with world size {misc.get_world_size()}")
    
    # Set seeds for reproducibility
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    if args.output_dir.strip():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Data augmentation and dataset creation
    transform_data = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6790435, 0.5052883, 0.66902906],
                             std=[0.19158737, 0.2039779, 0.15648715])
    ])
    dataset_train = ImageFolderInstance(args.data_path, transform=transform_data)
    dataset_val = ImageFolderInstance(args.data_val_path, transform=transform_data)
    print(dataset_train)
    
    # Use DistributedSampler in distributed mode
    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler_train = DistributedSampler(dataset_train, num_replicas=misc.get_world_size(), rank=args.rank, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=misc.get_world_size(), rank=args.rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir.strip() else None
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    ndata = len(dataset_train)
    
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m).to(device)
    criterion = NCECriterion(ndata).to(device)
    
    # Build the model identically on all ranks
    model = models_gcmae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        lemniscate=lemniscate,
        criterion=criterion,
        args=args
    ).to(device)
    
    # Debug: print total parameters on this rank
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Rank {args.rank if hasattr(args, 'rank') else 'Unknown'} total parameters: {total_params}")
    
    model_without_ddp = model
    print("Model =", model_without_ddp)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr:", args.lr * 256 / eff_batch_size)
    print("actual lr:", args.lr)
    print("accum iter:", args.accum_iter)
    print("effective batch size:", eff_batch_size)
    
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
        model_without_ddp = model.module
    
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr or args.blr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    nn_best = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)
    
        train_stats, nn_pred = train_one_epoch(
            model, data_loader_train, data_loader_val,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            lemniscate=lemniscate,
        )
    
        if nn_pred > nn_best:
            nn_best = nn_pred
    
        if args.output_dir.strip() and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
    
        if log_writer is not None:
            log_writer.flush()
    
        if args.output_dir.strip() and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, "epoch": epoch}
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    print("Training time", str(datetime.timedelta(seconds=int(total_time))))
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir.strip():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
