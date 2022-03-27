from pathlib import Path
import argparse

from torch import nn
from torchvision import datasets, transforms
import torch

import resnet
from timm.utils import accuracy, AverageMeter


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
        "--test-resize", default=256, type=int, help="validation data resize size"
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
        top1 = AverageMeter()
        top3 = AverageMeter()
        with torch.no_grad():
            for images, target in test_loader:
                output = model(images.to(device))
                acc1, acc3 = accuracy(
                        output, target.to(device), topk=(1, 3)
                )
                top1.update(acc1.item(), images.size(0))
                top3.update(acc3.item(), images.size(0))
        return top1, top3

    #Run inference
    top1, top3 = inference(model, test_loader, device)
    print(f"Inference results: Acc@1 {top1.avg:.3f}% | Acc@3 {top3.avg:.3f}%")


if __name__ == "__main__":
    main()
