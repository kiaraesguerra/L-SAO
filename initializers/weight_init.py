import torch.nn as nn

def OrthoInit(model, gain):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
    return model


def XavierInit(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model

def HeInit(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model

def UniformInit(model, a=0.0, b=1.0):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, a=a, b=b)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model