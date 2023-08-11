import argparse
import torch
from pytorch_lightning import Trainer


from dataloaders import get_dataloader
from callbacks import get_callback
from loggers import get_logger
from models import get_model
from initializers import get_initializer
from utils import remove_parameters, measure_sparsity
from train_utils import get_plmodule

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10")

parser.add_argument("--num-layers", type=int, default=5)
parser.add_argument("--width", "--hidden-width", type=int, default=16)
parser.add_argument("--weight-init", type=str, default="ortho")
parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])


parser.add_argument("--pruning-method", type=str, default=None)
parser.add_argument("--degree", type=int, default=None)
parser.add_argument("--sparsity", type=float, default=None)
parser.add_argument("--gain", type=float, default=1.0)


parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--image-size", type=int, default=32)
parser.add_argument("--autoaugment", type=bool, default=False)
parser.add_argument("--flip", type=bool, default=False)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", type=int, default=100, metavar="N")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--seed", type=int, default=3407)

parser.add_argument("--epochs", default=200, type=int, metavar="N")
parser.add_argument("--warmup-epoch", type=int, default=0)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--min-lr", default=1e-6, type=float)
parser.add_argument("--scheduler", type=str, default="multistep")
parser.add_argument("--criterion", default="crossentropy", type=str)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--nesterov", default=True, type=bool)
parser.add_argument("--milestones", default=[100, 150], type=list)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument(
    "--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamW"]
)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)
parser.add_argument("--eps", type=float, default=1.0e-08)
parser.add_argument("--amsgrad", type=bool, default=False)
parser.add_argument("--label-smoothing", type=float, default=0)

parser.add_argument("--device", type=str, default="cuda")


parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--ckpt-path-resume", type=str, default=None)
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--dirpath", type=str, default="results")
parser.add_argument("--filename", type=str, default="best")
parser.add_argument("--callbacks", type=list, default=["checkpoint"])
parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--save-last", type=bool, default=True)

args = parser.parse_args()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    train_dl, validate_dl, test_dl = get_dataloader(args)
    model = get_model(args)
    model = get_initializer(model, args)
    model = get_plmodule(model, args)
    callbacks = get_callback(args)
    logger = get_logger(args)
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_dl, validate_dl, ckpt_path=args.ckpt_path)
    trainer.test(dataloaders=test_dl, ckpt_path=args.ckpt_path)
    ckpt_path = callbacks[0].best_model_path
    model_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(model_checkpoint["state_dict"])


    remove_parameters(model)
    torch.save(
        model.state_dict(),
        f"{args.dirpath}/{args.experiment_name}/{args.experiment_name}.pt",
    )