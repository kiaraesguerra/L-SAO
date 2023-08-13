import torch
import torch.nn as nn
from ramanujan_constructions import Ramanujan_Constructions


class LowRankSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, mode='SAO', degree=64):
        super(LowRankSparseLinear, self).__init__()
        
        self.rank = rank
        self.mode = mode
        self.degree = degree
        self.in_features = in_features
        self.out_features = out_features
        self.W_layer = nn.Linear(in_features=in_features, out_features=rank, bias=False)
        self.U_layer = nn.Linear(in_features=rank, out_features=out_features, bias=False)
        self.S_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        
    def forward(self, x):
        x1 = self.W_layer(x)
        x1 = self.U_layer(x1)
        x2 = self.S_layer(x)
        return x1 + x2

class LowRankSparseInitializer:
    def __init__(self, model, mode, degree):
        self.mode = mode
        self.model = model
        self.degree = degree
        
    def sparse(self, module):
        if self.mode == 'SAO':
            constructor = Ramanujan_Constructions(height=module.out_features, width=module.in_features, degree=self.degree)
            sao_matrix, sao_mask = constructor()
            self.sao_matrix = sao_matrix
            self.sao_mask = sao_mask
            
        elif self.mode == 'LMP':
            s_module = nn.Linear(module.out_features, module.in_features).to('cuda')
            nn.init.orthogonal_(s_module.weight)
            torch.nn.utils.prune.ln_structured(s_module, name="weight", amount=0.5, n=2, dim=0)
            self.lmp_matrix = s_module.weight
            
    def low_rank(self, module):
        self.sparse(module)
    
        if self.mode == 'SAO':
            S_weight_matrix = self.sao_matrix
        elif self.mode == 'LMP':
            S_weight_matrix = self.lmp_matrix
        
        LR = module.weight.to('cuda') - S_weight_matrix.to('cuda')
        u, s, v = torch.linalg.svd(LR)
        s_diag = torch.diag_embed(s)
        rank = torch.sum(s > 1e-3)
        w = s_diag@v
        w_weight_matrix = w[0:rank, :] 
        u_weight_matrix = u[:, 0:rank] 
        LRS_module = LowRankSparseLinear(module.in_features, module.out_features,rank, self.degree)
        LRS_module.W_layer.weight = nn.Parameter(w_weight_matrix)
        LRS_module.U_layer.weight = nn.Parameter(u_weight_matrix)
        LRS_module.S_layer.weight = nn.Parameter(S_weight_matrix)
        torch.nn.utils.prune.custom_from_mask(LRS_module.S_layer, name="weight", mask=(S_weight_matrix!=0)*1)
   
        return LRS_module

    def initialize_low_rank(self):

        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
            
        for module_name, module in self.model.hidden_layers.named_modules():
            if isinstance(module, nn.Linear):
                self.model.hidden_layers._modules[module_name] = self.low_rank(module)
            
        return self.model
    
def LowRankSparse(model, mode, degree):
    initializer = LowRankSparseInitializer(model, mode=mode, degree=degree)
    initialized_model = initializer.initialize_low_rank().to('cuda')
    return initialized_model