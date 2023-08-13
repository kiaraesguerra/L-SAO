from initializers.weight_init import OrthoInit, XavierInit, HeInit, UniformInit
import torch.nn as nn


def get_weight_init(model, args):
    if args.weight_init == 'xavier':
        model = XavierInit(model)
    elif args.weight_init == 'he':
        model = HeInit(model)
    elif args.weight_init == 'uniform':
        model = UniformInit(model, a=args.a, b=args.b)
    elif args.weight_init == 'ortho':
        model = OrthoInit(model, gain=args.gain)
        
    return model
        
    