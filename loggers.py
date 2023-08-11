from pytorch_lightning.loggers import CSVLogger


def get_logger(args):
    return CSVLogger(f"{args.dirpath}/{args.experiment_name}", name="logs")
