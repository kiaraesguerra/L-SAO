from models.mlp import MLP
from models.mixer import MLPMixer
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.trial1 import trial1

def get_model(args):
    if args.model == "mlp":
        model = MLP(
            image_size=args.image_size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            activation=args.activation,
            num_layers=args.num_layers,
            hidden_width=args.width,
        )
    
    elif args.model == "mixer":
        model = MLPMixer(
            in_channels=3,
            img_size=args.image_size,
            patch_size=4,
            hidden_size=128,
            hidden_s=64,
            hidden_c=512,
            num_layers=8,
            num_classes=args.num_classes,
            drop_p=0.0,
            off_act=False,
            is_cls_token=True,
        ).to("cuda")
        
    elif args.model == "trial1":
        model = trial1().to('cuda')
    elif args.model == "resnet20":
        model = resnet20()
    elif args.model == "resnet32":
        model = resnet32()
    elif args.model == "resnet44":
        model = resnet44()
    elif args.model == "resnet56":
        model = resnet56()

    return model
