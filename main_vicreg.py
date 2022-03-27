# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import logging
import warnings

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from timm.utils import AverageMeter, setup_default_logging

import augmentations as aug

import utils
import resnet
from deit_models import deit
from randoms import set_seed, set_worker_seed

warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    parser.add_argument(
        "--seed", default=7, type=int, help="seed for reproducibility"
    )
    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=25,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--norm-weight-decay",
                        default=None,
                        type=float,
                        help="weight decay for Normalization layers (default: None, same value as --wd)", )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int,
                        help="the number of epochs to warmup")
    # EMA
    parser.add_argument("--model-ema", action="store_true",
                        help="enable Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32,
                        help="number of iterations for updating EMA model", )
    parser.add_argument("--model-ema-decay", type=float, default=0.99998,
                        help="Exponential Moving Average decay factor")

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    #Aug
    parser.add_argument("--aug", action="store_true", help="enable auto data augmentation")
    parser.add_argument(
        "--train-crop", default=224, type=int, help="training data random crop size"
    )
    # Running
    parser.add_argument("--num-workers", type=int, default=8)

    return parser


def main(args):
    set_seed(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    log_file = args.exp_dir / "log.txt"

    setup_default_logging(log_path=log_file)
    _logger = logging.getLogger('Pre-training')
    _logger.info(args)
    _logger.info(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transforms = aug.TrainTransform(args.train_crop)
    if args.aug:
        transforms = aug.StrongTrainTransform(args.train_crop)

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=g,
    )

    if 'deit' in args.arch:
        backbone, embedding = deit(args.arch)
        norm_shape = backbone[2][0].norm1.normalized_shape[0]
        args.mlp = "{norm}-{norm}-{norm}".format(norm=norm_shape)
    else:
        backbone, embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

    model = VICReg(args, backbone, embedding).to(device)

    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.wd]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    optimizer = LARS(
        parameters,
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    model_ema = None
    if args.model_ema:
        adjust = args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    if (args.exp_dir / "model.pth").is_file():
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if model_ema:
            model_ema.load_state_dict(ckpt["model_ema"])
        _logger.info(f"Resuming from epoch {start_epoch} ...")
    else:
        start_epoch = 0

    def train_one_epoch(args, epoch, model, loader, device, optimizer, stats_file, model_ema=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        model.train()

        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            start_time = time.time()
            x = x.to(device)
            y = y.to(device)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            loss = model.forward(x, y)
            loss.backward()
            optimizer.step()

            if model_ema and step % args.model_ema_steps == 0:
                _logger.info(f"Updating EMA model params")
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)

            losses.update(loss.item(), x.size(0))

            end_time = time.time()
            batch_time.update(end_time - start_time)

            if step % args.log_freq_time == 0:
                _logger.info(
                    'Training --> Epoch: {} | Step: [{}/{} ({:>3.0f}%)] | '
                    'Loss: {loss.val:.3f} ({loss.avg:.3f}) | '
                    'Time: {batch_time.val:.3f}s, {rate:.3f}/s ({batch_time.avg:.3f}s, {rate_avg:.3f}/s) | '
                    'lr: {lr:.3e}'.format(epoch, step, len(loader) * args.epochs,
                                          100. * step / (len(loader) * args.epochs),
                                          loss=losses,
                                          batch_time=batch_time,
                                          rate=x.size(0) / batch_time.val,
                                          rate_avg=x.size(0) / batch_time.avg,
                                          lr=lr
                                          )
                )
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(end_time - start_time),
                    lr=lr,
                )

                print(json.dumps(stats), file=stats_file)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(args, epoch, model, loader, device, optimizer, stats_file, model_ema)

        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if model_ema:
            state["model_ema"] = model_ema.state_dict()
        torch.save(state, args.exp_dir / "model.pth")

    torch.save(model.backbone.state_dict(), args.exp_dir / "resnet50.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args, backbone, embedding):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone = backbone
        self.embedding = embedding
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
                self.args.sim_coeff * repr_loss
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
            self,
            params,
            lr,
            weight_decay=0,
            momentum=0.9,
            eta=0.001,
            weight_decay_filter=None,
            lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
