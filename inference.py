from pathlib import Path
import argparse

from torch import nn
from torchvision import datasets, transforms
import torch
import resnet


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Performing inference with a fine-tuned model"
    )
    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")

    # Checkpoint
    parser.add_argument("--finetuned", type=Path, help="path to pretrained model")
    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    # Running
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    # Data Augmentation
    parser.add_argument(
        "--test-resize", default=240, type=int, help="validation data resize size"
    )
    parser.add_argument(
        "--test-crop", default=224, type=int, help="validation data centre crop size"
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data loading code
    testdir = args.data_dir / "test"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose(
            [
                transforms.Resize(args.test_resize),
                transforms.CenterCrop(args.test_crop),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
            ]
        ),
    )

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, **kwargs)

    # Model definition and loading
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    head = nn.Linear(embedding, len(test_dataset.classes))
    model = nn.Sequential(backbone, head)
    state_dict = torch.load(args.finetuned, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    def inference(model, test_loader, device):
        print(f"Running inference with {args.arch} on full test split...")
        model.eval()
        top1 = AverageMeter("Acc@1", ":.3f")
        top3 = AverageMeter("Acc@3", ":.3f")
        with torch.no_grad():
            for images, target in test_loader:
                output = model(images.to(device))
                acc1, acc3 = accuracy(
                        output, target.to(device), topk=(1, 3)
                )
                top1.update(acc1[0].item(), images.size(0))
                top3.update(acc3[0].item(), images.size(0))
        return top1, top3

    #Run inference
    top1, top3 = inference(model, test_loader, device)
    print(f"Inference results: {top1.avg:.3f}% | {top3.avg:.3f}%")

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
