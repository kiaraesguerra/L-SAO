import warmup_scheduler
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, args):
    scheduler_dict = {
        "multistep": lambda: lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        ),
        "cosine": lambda: lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        ),
        "lambda": lambda: lr_scheduler.LambdaLR(optimizer),
        "plateau": lambda: lr_scheduler.ReduceLROnPlateau(optimizer),
        "cyclic": lambda: lr_scheduler.CyclicLR(optimizer),
    }

    base_scheduler = scheduler_dict.get(args.scheduler, lambda: None)()

    if args.warmup_epoch:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.warmup_epoch,
            after_scheduler=base_scheduler,
        )
    else:
        scheduler = base_scheduler

    return scheduler
