import torch.nn as nn


def get_criterion(args):
    criterion_mapping = {
        "crossentropy": nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        "bce": nn.BCELoss(),
        "l1": nn.L1Loss(),
    }

    criterion = criterion_mapping.get(args.criterion.lower(), None)
    return criterion
