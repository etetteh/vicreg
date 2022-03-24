# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import sys
import time

from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import InterpolationMode
import torch

import utils
import resnet
import create_subset
from randoms import set_seed

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model"
    )
    parser.add_argument(
        "--seed", default=7, type=int, help="seed for reproducibility"
    )
    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        help="size of training set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup"
    )
    parser.add_argument(
        "--lr-warmup-decay",
        default=0.01,
        type=float,
        help="the decay for lr"
    )
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    # Label smoothing
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing",
        dest="label_smoothing"
    )
    # EMA
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="number of iterations for updating EMA model",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="Exponential Moving Average decay factor"
    )
    # Data Augmentation
    parser.add_argument(
        "--val-resize", default=256, type=int, help="validation data resize size"
    )
    parser.add_argument(
        "--val-crop", default=224, type=int, help="validation data centre crop size"
    )
    parser.add_argument(
        "--train-crop", default=224, type=int, help="training data random crop size"
    )
    parser.add_argument("--auto-augment", default=None, type=str, choices=("trivial", "rand"), help="auto augment policy")

    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability")
    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    if args.train_percent != 100:
        if (args.data_dir / f'{args.train_percent}percent_train_subset.txt').is_file():
            print(f"Loading {args.train_percent} percent train subset images names...")
            args.train_files = open(args.data_dir/f'{args.train_percent}percent_train_subset.txt', 'r').readlines()
        else:
            print(f"Creating {args.train_percent} percent train subset images names...")
            create_subset.create_data_subset(dataset_path=args.data_dir, subset_percent=args.train_percent)
            args.train_files = open(args.data_dir/f'{args.train_percent}percent_train_subset.txt', 'r').readlines()
    main_worker(args)


def main_worker(args):
    set_seed(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data loading code
    traindir = args.data_dir / "train"
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transforms = [
            transforms.RandomResizedCrop(args.train_crop, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip()
    ]
    if args.auto_augment == "trivial":
        train_transforms.append(autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))
    elif args.auto_augment == "rand":
        train_transforms.append(autoaugment.RandAugment(interpolation=InterpolationMode.BILINEAR))

    train_transforms.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ]
    )
    if args.random_erase > 0:
        train_transforms.append(transforms.RandomErasing(p=args.random_erase))

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(train_transforms)
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(args.val_resize),
                transforms.CenterCrop(args.val_crop),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
            ]
        ),
    )

    if args.train_percent != 100:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.strip()
            cls = fname.split("_")[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls])
            )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, **kwargs)

    # Model definition and loading
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    assert missing_keys == [] and unexpected_keys == []

    head = nn.Linear(embedding, len(train_dataset.classes))
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    model.to(device)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))

    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs)

    if args.lr_warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    model_ema = None
    if args.model_ema:
        adjust = args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if model_ema:
            model_ema.load_state_dict(ckpt["model_ema"])
        print(f"Resume training from epoch {start_epoch}...")
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top3=0)

    def train_one_epoch(args, epoch, model, train_loader, device, criterion, optimizer, stats_file, model_ema=None,):
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False
        start_time = time.time()
        for step, (images, target) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            output = model(images.to(device))
            loss = criterion(output, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_ema and step % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)

            if step % args.print_freq == 0:
                pg = optimizer.param_groups
                lr_head = pg[0]["lr"]
                lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                stats = dict(
                    epoch=epoch,
                    step=step,
                    lr_backbone=lr_backbone,
                    lr_head=lr_head,
                    loss=loss.item(),
                    time=time.strftime("%M:%S", time.gmtime(int(time.time() - start_time))),
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

    def evaluate(epoch, model, val_loader, device, stats_file, log_suffix=""):
        model.eval()
        top1 = AverageMeter("Acc@1")
        top3 = AverageMeter("Acc@3")
        with torch.no_grad():
            for images, target in val_loader:
                output = model(images.to(device))
                acc1, acc3 = accuracy(
                        output, target.to(device), topk=(1, 3)
                )
                top1.update(acc1[0].item(), images.size(0))
                top3.update(acc3[0].item(), images.size(0))

        if top1.avg > best_acc.top1:
            print(f"Acc@1 improved from {best_acc.top1:.3f}% to {top1.avg:.3f}%. Saving model state... ")
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top3 = max(best_acc.top3, top3.avg)
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.exp_dir / f"best_model_epoch-{epoch}.pth")

        stats = dict(
            log_suffix=log_suffix,
            epoch=epoch,
            acc1=top1.avg,
            acc3=top3.avg,
            best_acc1=best_acc.top1,
            best_acc3=best_acc.top3,
        )

        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(args, epoch, model, train_loader, device, criterion, optimizer, stats_file, model_ema)
        scheduler.step()
        evaluate(epoch, model, val_loader, device, stats_file)
        if model_ema:
            evaluate(epoch, model_ema, val_loader, device, stats_file, log_suffix="EMA")

        state = dict(
            epoch=epoch + 1,
            best_acc=best_acc,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )
        if model_ema:
            state["model_ema"] = model_ema.state_dict()
        torch.save(state, args.exp_dir / "checkpoint.pth")


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
