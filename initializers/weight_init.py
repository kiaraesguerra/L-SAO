import torch.nn as nn
import torch


def _ortho_gen(rows, columns) -> torch.tensor:
    rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
    q, _ = torch.qr(rand_matrix)
    orthogonal_matrix = q[:, :columns]
    return orthogonal_matrix.T if columns > rows else orthogonal_matrix


def _concat(matrix):
    W = torch.concat(
        [
            torch.concat([matrix, torch.negative(matrix)], axis=0),
            torch.concat([torch.negative(matrix), matrix], axis=0),
        ],
        axis=1,
    )
    return W


def _ortho_generator(module, activation) -> torch.tensor:
    if activation == "relu":
        rows = module.out_features // 2
        columns = module.in_features // 2
        orthogonal_matrix = _concat(_ortho_gen(rows, columns))

    else:
        rows = module.out_features
        columns = module.in_features
        orthogonal_matrix = _ortho_gen(rows, columns)
    return orthogonal_matrix


def OrthoInit(model, gain, activation):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(_ortho_generator(module, activation) * gain)

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
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
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
