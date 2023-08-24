from models.mlp import MLP
from models.mixer import MLPMixer
from models.residualmlp import ResidualMLP

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
    elif args.model == "residualmlp":
        model = ResidualMLP(
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

    return model
