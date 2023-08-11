import sys
import torch
import torchvision
import torchvision.transforms as transforms
import os

from AutoAugment.autoaugment import CIFAR10Policy, SVHNPolicy


def get_dataloader(args):
    train_transform, test_transform = get_transform(args)

    if args.dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            "./datasets", train=True, transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.CIFAR10(
            "./datasets", train=False, transform=test_transform, download=True
        )
        validate_ds = None
        args.num_classes = 10
        args.in_channels = 3

    elif args.dataset == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            "./datasets", train=True, transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.CIFAR100(
            "./datasets", train=False, transform=test_transform, download=True
        )
        validate_ds = None
        args.num_classes = 100
        args.in_channels = 3

    elif args.dataset == "svhn":
        train_ds = torchvision.datasets.SVHN(
            "./datasets", split="train", transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.SVHN(
            "./datasets", split="test", transform=test_transform, download=True
        )
        validate_ds = None
        args.num_classes = 10
        args.in_channels = 3
        
    elif args.dataset == "mnist":
        train_ds = torchvision.datasets.MNIST(
            "./datasets", split="train", transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.MNIST(
            "./datasets", split="test", transform=test_transform, download=True
        )
        validate_ds = None
        args.num_classes = 10
        args.in_channels = 1       
        

    elif args.dataset == "cinic10":
        dir = "../data/cinic-10"

        traindir = os.path.join(dir, "train")
        validatedir = os.path.join(dir, "valid")
        testdir = os.path.join(dir, "test")
        train_ds = torchvision.datasets.ImageFolder(
            root=traindir, transform=train_transform
        )
        validate_ds = torchvision.datasets.ImageFolder(
            root=validatedir, transform=test_transform
        )
        test_ds = torchvision.datasets.ImageFolder(
            root=testdir, transform=test_transform
        )
        args.num_classes = 10
        args.in_channels = 3
    else:
        raise ValueError(f"No such dataset:{args.dataset}")

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if validate_ds:
        validate_dl = torch.utils.data.DataLoader(
            validate_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        validate_dl = test_dl

    return train_dl, validate_dl, test_dl


def get_transform(args):
    if args.dataset in ["cifar10", "cifar100", "svhn", "cinic10"]:
        args.padding = 4
        args.size = 32
        if args.dataset == "cifar10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif args.dataset == "cifar100":
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        elif args.dataset == "svhn":
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        elif args.dataset == "cinic10":
            args.mean, args.std = [0.47889522, 0.47227842, 0.43047404], [
                0.24205776,
                0.23828046,
                0.25874835,
            ]

    else:
        args.padding = 28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform_list = [
        transforms.RandomCrop(size=(args.size, args.size), padding=args.padding),
        transforms.Resize(size=(args.image_size, args.image_size)),
    ]

    if args.flip:
        train_transform_list.extend([transforms.RandomHorizontalFlip()])

    if args.autoaugment:
        if (
            args.dataset in ["cifar10", "cifar100", "cinic10"]
        ):
            train_transform_list.append(CIFAR10Policy())
        elif args.dataset == "svhn":
            train_transform_list.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")

    train_transform = transforms.Compose(
        train_transform_list
        + [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(size=(args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )

    return train_transform, test_transform
