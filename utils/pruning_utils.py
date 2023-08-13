import torch
import torch.nn.utils.prune as prune


def measure_sparsity(model):
    num_zeros = torch.tensor(0).to("cuda")
    num_elements = torch.tensor(0).to("cuda")

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros = torch.sum(module.weight == 0)
            module_num_elements = module.weight.nelement()
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):
            module_num_zeros = torch.sum(module.weight == 0)
            module_num_elements = module.weight.nelement()
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements
    return sparsity


def remove_parameters(model):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model
