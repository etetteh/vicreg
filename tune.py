from __future__ import print_function, division

import os
import random
import argparse
import numpy as np
from functools import partial
from pathlib import Path

import timm
import torch, torchvision

from torch import nn
from timm.loss import BinaryCrossEntropy, LabelSmoothingCrossEntropy

from torchvision import datasets, models
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import utils
import resnet
import create_subset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, pb2



def load_data(config, data_dir=''):
    normalize = transforms.Normalize(
        mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
    )
    augment = [transforms.RandomResizedCrop(size=224, interpolation=InterpolationMode.BILINEAR),
               transforms.RandomHorizontalFlip()]

    ######### Recipe 2 #########
    if cfg.random:
        augment.append(autoaugment.RandAugment(num_ops=config["num_ops"], magnitude=config["magnitude"], num_magnitude_bins=config["num_magnitude_bins"]))
    elif cfg.num_ops:
        augment.append(autoaugment.RandAugment(num_ops=cfg.num_ops, magnitude=cfg.magnitude, num_magnitude_bins=cfg.num_magnitude_bins, interpolation=InterpolationMode(cfg.interpolate)))
        
    if cfg.trivial:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=config["num_magnitude_bins"]))
    elif cfg.num_magnitude_bins:
        augment.append(autoaugment.TrivialAugmentWide(num_magnitude_bins=cfg.num_magnitude_bins, interpolation=InterpolationMode(cfg.interpolate)))

    augment.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
    ])

    ######### Recipe 4 #########
    # if cfg.random_erase:
    #     augment.append(torchvision.transforms.RandomErasing(p=config["random_erase_prob"]))
    # elif cfg.random_erase_prob:
    #     augment.append(torchvision.transforms.RandomErasing(p=cfg.random_erase_prob))
    
    ######### Recipe 9 #########
    if cfg.fixres:
        augment[0] = transforms.RandomResizedCrop(size=config["train_crop_size"], interpolation=InterpolationMode(cfg.interpolate))
    if cfg.train_crop:
        augment[0] = transforms.RandomResizedCrop(size=cfg.train_crop, interpolation=InterpolationMode(cfg.interpolate))

    augment = transforms.Compose(augment)
    
    valid_augment = transforms.Compose([
            transforms.Resize(cfg.val_resize),
            transforms.CenterCrop(cfg.val_crop),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize
            ])

    if cfg.infer_resize:
        valid_augment[0] = transforms.Resize(config["infer_resize"])
                       
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=augment)
    validset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=valid_augment)
    num_classes = len(trainset.classes)
    return trainset, validset, num_classes
    
    
def train(config, pretrained='', checkpoint_dir=None, data_dir=None):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    trainset, validset, num_classes = load_data(config, data_dir=data_dir)
    
    if cfg.train_percent != 100:
        if os.path.isfile(data_dir + f'/{cfg.train_percent}percent_train_subset.txt'):
            print(f"Loading {cfg.train_percent} percent train subset images names...")
            cfg.train_files = open(data_dir + f'/{cfg.train_percent}percent_train_subset.txt', 'r').readlines()
        else:
            print(f"Creating {cfg.train_percent} percent train subset images names...")
            create_subset.create_data_subset(dataset_path=data_dir, subset_percent=cfg.train_percent)
            cfg.train_files = open(data_dir + f'/{cfg.train_percent}percent_train_subset.txt', 'r').readlines()
            
    if cfg.train_percent != 100:
        trainset.samples = []
        for fname in cfg.train_files:
            fname = fname.strip()
            cls = fname.split("_")[0]
            trainset.samples.append(
                (data_dir + '/train/' + cls + "/" + fname, trainset.class_to_idx[cls])
            )

    backbone, embedding = resnet.__dict__[cfg.model](zero_init_residual=True)
    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == []

    head = nn.Linear(embedding, len(trainset.classes))
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)

    if cfg.lr_head:
        parameters = [dict(params=head.parameters(), lr=cfg.lr_head)]
    else:
        parameters = [dict(params=head.parameters(), lr=config["lr_head"])]
        
    if cfg.weights == "finetune":
        if cfg.lr_backbone:
            parameters.append(dict(params=backbone.parameters(), lr=cfg.lr_backbone))
        else:
            parameters.append(dict(params=backbone.parameters(), lr=config["lr_backbone"]))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    if cfg.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    
    ######### Recipe 10 #########
    ema_model = None
    if cfg.ema:
        adjust = 1 * cfg.batch_size * config["model_ema_steps"] / cfg.epochs
        alpha = 1.0 - config["model_ema_decay"]
        alpha = min(1.0, alpha * adjust)
        ema_model = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    elif (cfg.model_ema_decay and cfg.model_ema_steps):
        adjust = 1 * cfg.batch_size * cfg.model_ema_steps / cfg.epochs
        alpha = 1.0 - cfg.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        ema_model = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    ######### Recipe 1 #########
    if cfg.lr_optim:
        optimizer = torch.optim.AdamW(parameters, lr=0, weight_decay=config["weight_decay"])
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - config["lr_warmup_epochs"])
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=config["lr_warmup_decay"], total_iters=config["lr_warmup_epochs"])
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config["lr_warmup_epochs"]])
    elif cfg.lr_head:
        optimizer = torch.optim.AdamW(parameters, lr=0, weight_decay=cfg.weight_decay)
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs)
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=cfg.lr_warmup_decay, total_iters=cfg.lr_warmup_epochs)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[cfg.lr_warmup_epochs])
        
    ######### Recipe 5 #########
    if cfg.bce:
        train_criterion = BinaryCrossEntropy(smoothing=0.0)
        if cfg.smooth:
            train_criterion = BinaryCrossEntropy(smoothing=cfg.smooth)
        elif cfg.label_smoothing:
            train_criterion = BinaryCrossEntropy(smoothing=config["label_smoothing"])
    else:
        train_criterion = nn.CrossEntropyLoss()
        if cfg.smooth:
            train_criterion = LabelSmoothingCrossEntropy(smoothing=cfg.smooth)
        elif cfg.label_smoothing:
            train_criterion = LabelSmoothingCrossEntropy(smoothing=config["label_smoothing"])

    train_criterion = train_criterion.to(device)
    valid_criterion = nn.CrossEntropyLoss().to(device)
        
    if checkpoint_dir:
        checkpoints = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(checkpoints["model_state"])
        optimizer.load_state_dict(checkpoints["optimizer_state"])
        lr_scheduler.load_state_dict(checkpoints["lr_scheduler_state"])
        if ema_model:
            ema_model.load_state_dict(checkpoints["model_ema"])

    train_sampler = torch.utils.data.RandomSampler(trainset)
    val_sampler = torch.utils.data.SequentialSampler(validset)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size if cfg.batch_size is not None else int(config["batch_size"]),
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=cfg.val_batch_size,
        sampler=val_sampler,
        num_workers=cfg.workers,
        pin_memory=True
    )

    for epoch in range(cfg.epochs):
        train_one_epoch(config, epoch, train_loader, model, ema_model, optimizer, train_criterion, device=device)

        lr_scheduler.step()
        if ema_model:
            valid_loss, accuracy = validate(valid_loader, ema_model, valid_criterion, device=device)
        else:
            valid_loss, accuracy = validate(valid_loader, model, valid_criterion, device=device)
                
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            ckpts = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict()
            }
            if ema_model:
                ckpts["ema_state"] = ema_model.state_dict()
            torch.save(ckpts, path)
            
        tune.report(loss=valid_loss, accuracy=accuracy)
    print("Finished Training")
    

def train_one_epoch(config, epoch, train_loader, model, ema_model, optimizer, criterion, device="cpu"):
    if cfg.weights == "finetune":
        model.train()
    elif cfg.weights == "freeze":
        model.eval()
    else:
        assert False
        
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
            running_loss = 0.0

        if cfg.ema:
            if ema_model and i % config["model_ema_steps"] == 0:
                ema_model.update_parameters(model)
                if epoch < cfg.lr_warmup_epochs:
                    ema_model.n_averaged.fill_(0)
        elif cfg.model_ema_decay and cfg.model_ema_steps:
                if ema_model and i % cfg.model_ema_steps == 0:
                    ema_model.update_parameters(model)
                    if epoch < cfg.lr_warmup_epochs:
                        ema_model.n_averaged.fill_(0)
        


def validate(valid_loader, model, criterion, device="cpu"):
    valid_loss = 0.0
    valid_steps = 0
    total = 0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            valid_loss += loss.cpu().numpy()
            valid_steps += 1

    valid_loss = valid_loss / valid_steps
    accuracy = correct / total
    return valid_loss, accuracy
   

def main(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    data_dir = os.path.abspath(cfg.data_dir)
    pretrained = ""
    if cfg.pretrained is not None:
        pretrained = os.path.abspath(cfg.pretrained)
    
    config = {}
    hyperparam_mutations = {}
    
    ######### Recipe 1 #########
    ### lr optimization ###
    if cfg.lr_optim:
        config["lr_head"] = tune.qloguniform(1e-4, 1e-1, 1e-5)
        config["lr_warmup_decay"] = tune.qloguniform(1e-5, 1e-3, 1e-6)
        config["lr_warmup_epochs"] = tune.qrandint(3, 15, 3)
        config["batch_size"] = tune.grid_search([16, 32, 64, 128])
        config["weight_decay"] = tune.qloguniform(1e-5, 1e-3, 1e-6)
        
        hyperparam_mutations["lr_head"] = [1e-4, 1e-1]
        hyperparam_mutations["lr_warmup_decay"] = [1e-5, 1e-3]
        hyperparam_mutations["lr_warmup_epochs"] = [3, 15]
        hyperparam_mutations["batch_size"] = [16, 128]
        hyperparam_mutations["weight_decay"] = [1e-5, 1e-3]
        
        if cfg.weights == "finetune":
            config["lr_backbone"] = tune.qloguniform(1e-4, 1e-1, 1e-5)
            hyperparam_mutations["lr_backbone"] = [1e-4, 1e-1]

    ######### Recipe 2 #########
    if cfg.trivial:
        config["num_magnitude_bins"] = tune.randint(15, 35)
        
        hyperparam_mutations["num_magnitude_bins"] = [15, 35]
        
    if cfg.random:
        config["num_ops"] = tune.randint(2, 5)
        config["magnitude"] = tune.randint(5, 9)
        config["num_magnitude_bins"] = tune.randint(15, 35)
        
        hyperparam_mutations["num_ops"] = [2, 5]
        hyperparam_mutations["magnitude"] = [5, 9]
        hyperparam_mutations["num_magnitude_bins"] = [15, 35]

    ######### Recipe 4 #########  
    ### data aug ###
    # if cfg.random_erase:
    #     config["random_erase_prob"] = tune.uniform(0.2, 0.3)
    #
    #     hyperparam_mutations["random_erase_prob"] = [0.1, 0.3]

    ######### Recipe 5 ######### 
    ### label smoothing
    if cfg.label_smoothing:
        config["label_smoothing"] = tune.grid_search([0.05, 0.1, 0.15])
        
        hyperparam_mutations["label_smoothing"] = [0.05, 0.15]

    ######### Recipe 9 #########   
    ### fixres ###
    if cfg.fixres:
        config["train_crop_size"] = tune.grid_search([168, 176, 184, 192, 200, 208, 216, 224])
        
        hyperparam_mutations["train_crop_size"] =[168, 224]

    ######### Recipe 10 #########
    ### model ema ###
    if cfg.ema:
        config["model_ema_steps"] = tune.qrandint(8, 48, 8)
        config["model_ema_decay"] = tune.uniform(0.99, 0.99998)
        
        hyperparam_mutations["model_ema_steps"] = [8, 48]
        hyperparam_mutations["model_ema_decay"] = [0.99, 0.99998]

    if cfg.infer_resize:
        config["infer_resize"] = tune.grid_search([224, 232, 240, 248, 256])
        
    print(config)
    load_data(config, data_dir)
    
    if cfg.asha:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            max_t=cfg.epochs,
            grace_period=1,
            reduction_factor=2
        )
    
    if cfg.pbt:
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            perturbation_interval=300.0,
            hyperparam_mutations=hyperparam_mutations
        )
        
    if cfg.pb2:
        scheduler = pb2.PB2(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            perturbation_interval=300.0,
            hyperparam_bounds=hyperparam_mutations
        )
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
        
    result = tune.run(
        partial(train, data_dir=data_dir, pretrained=pretrained),
        name=cfg.name,
        resources_per_trial={"cpu": cfg.cpus_per_trial, "gpu": cfg.gpus_per_trial},
        config=config,
        num_samples=cfg.num_samples,
        scheduler=scheduler,
        stop={"accuracy": 0.99},
        resume="AUTO",
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    return result    
    

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Model Optimization Training", add_help=add_help)
    parser.add_argument("--data_dir", default="", type=Path, help="dataset path")
    parser.add_argument("--name", default="hparams_tune", type=str, help="")
    parser.add_argument("--seed", default=99, type=int, help="")
    parser.add_argument("--workers", default=8, type=int, metavar="N", help="",)
    parser.add_argument("--model", default="resnet50", type=str, help="")
    
    parser.add_argument("--num_samples", default=3, type=int, help="")
    parser.add_argument("--epochs", default=5, type=int, help="")
    parser.add_argument("--cpus_per_trial", default=1, type=int, help="")
    parser.add_argument("--gpus_per_trial", default=0, type=int, help="")
    
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"), help="",)
    parser.add_argument("--pretrained", type=Path, help="")
    parser.add_argument("--train_percent", default=100, type=int, help="",)
        
    ###### scheduler type
    parser.add_argument("--asha", action="store_true", default=False, help="")
    parser.add_argument("--pbt", action="store_true", default=False, help="")
    parser.add_argument("--pb2", action="store_true", default=False, help="")
    
    ###### pass to instantiate recipe                   
    parser.add_argument("--lr_optim", action="store_true", default=False, help="")
    parser.add_argument("--trivial", action="store_true", default=False, help="")
    parser.add_argument("--random", action="store_true", default=False, help="")
    parser.add_argument("--random_erase", action="store_true", default=False, help="")
    parser.add_argument("--label_smoothing", action="store_true", default=False, help="")
    parser.add_argument("--fixres", action="store_true", default=False, help="")
    parser.add_argument("--infer_resize", action="store_true", default=False, help="")
    parser.add_argument("--ema", action="store_true", default=False, help="")
    parser.add_argument("--bce", action="store_true", default=False, help="")
    
    ###### pass to use optimized hparam                 
    parser.add_argument("--batch_size", default=None, type=int, help="")
    parser.add_argument("--val_batch_size", default=32, type=int, help="")
    parser.add_argument("--lr_backbone", default=None, type=float, metavar="LR", help="",)
    parser.add_argument("--lr_head", default=None, type=float, metavar="LR", help="",)
    parser.add_argument("--lr_warmup_epochs", default=None, type=int, help="")
    parser.add_argument("--lr_warmup_decay", default=None, type=float, help="")
    parser.add_argument("--weight_decay", default=None, type=float, help="")
    parser.add_argument("--smooth", default=None, type=float, help="")
    parser.add_argument("--random_erase_prob", default=None, type=float, help="")
    parser.add_argument("--model_ema_steps", default=None, type=int, help="")
    parser.add_argument("--model_ema_decay", default=None, type=float, help="")
    parser.add_argument("--train_crop", default=224, type=int, help="")
    parser.add_argument("--val_crop", default=224, type=int, help="")
    parser.add_argument("--val_resize", default=256, type=int, help="")
    parser.add_argument("--interpolate", default="bilinear", type=str, help="")
    parser.add_argument("--num_ops", default=None, type=int, help="")
    parser.add_argument("--magnitude", default=None, type=int, help="")
    parser.add_argument("--num_magnitude_bins", default=None, type=int, help="")
    # parser.add_argument("--", default=None, type=int, help="")
    
    return parser
    
    
if __name__ == "__main__":
    cfg = get_args_parser().parse_args()

    ## seed for ray[tune] schedulers
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # You can change the number of GPUs per trial here:
    result = main(cfg)
