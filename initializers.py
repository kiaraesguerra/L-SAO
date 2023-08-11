from lowrank import LowRankSparse
import torch.nn as nn

def OrthoInit(model, gain):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
    return model

def get_initializer(model, args):
    if args.weight_init == 'LSAO':
        model = LowRankSparse(model, mode='SAO', degree=args.degree)
    elif args.weight_init == 'ortho':
        model = OrthoInit(model, args.gain)
    elif args.weight_init == 'LS':
        model = LowRankSparse(model, mode='LMP', degree=args.degree)
        
    return model
        
        
        
        
    