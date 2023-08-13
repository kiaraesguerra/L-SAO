from mlp import MLP

def get_model(args):
    model = MLP(
            image_size=args.size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            activation=args.activation,
            num_layers=args.num_layers,
            hidden_width=args.width,
        )
    return model