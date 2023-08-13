import torch
import torch.nn as nn
from ls_initializers.ramanujan_constructions import Ramanujan_Constructions


class LSModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        sparse_matrix: str = None,
        sparsity: float = None,
        degree: float = None,
    ):
        super(LSModule, self).__init__()

        self.rank = rank
        self.sparse_matrix = sparse_matrix
        self.degree = degree
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features
        self.W_layer = nn.Linear(in_features=in_features, out_features=rank, bias=False)
        self.U_layer = nn.Linear(
            in_features=rank, out_features=out_features, bias=False
        )
        if self.sparse_matrix:
            self.S_layer = nn.Linear(
                in_features=in_features, out_features=out_features, bias=False
            )

    def forward(self, x):
        x1 = self.W_layer(x)
        out = self.U_layer(x1)
        if self.sparse_matrix:
            x2 = self.S_layer(x)
            out += x2
        return out


class LowRankSparseInitializer:
    def __init__(
        self,
        model: nn.Module,
        sparse_matrix: str = None,
        sparsity: float = None,
        degree: int = None,
        activation: str = 'tanh'
    ):
        self.sparse_matrix = sparse_matrix
        self.model = model
        self.sparsity = sparsity
        self.degree = degree
        self.activation = activation

    def _sparse_matrix(self, module):
        if self.sparse_matrix in ["SAO", "RG-N", "RG-U"]:
            constructor = Ramanujan_Constructions(
                height=module.out_features,
                width=module.in_features,
                method=self.sparse_matrix,
                sparsity=self.sparsity,
                degree=self.degree,
                activation=self.activation
            )
            s_weight_matrix, _ = constructor()

        elif self.sparse_matrix == "LMP":
            s_module = nn.Linear(module.out_features, module.in_features).to("cuda")
            nn.init.orthogonal_(s_module.weight)
            torch.nn.utils.prune.ln_structured(
                s_module, name="weight", amount=self.sparsity, n=2, dim=0
            )
            s_weight_matrix = s_module.weight

        return s_weight_matrix

    def _low_rank_sparse(self, module):
        # Acquiring the original weight matrix
        LR = module.weight.to("cuda")

        # Acquiring the sparse matrix and subtracting it from the original weight matrix
        if self.sparse_matrix:
            s_weight_matrix = self._sparse_matrix(module)
            LR = LR - s_weight_matrix

        # Performing SVD on the difference matrix
        u, s, v = torch.linalg.svd(LR)
        s_diag = torch.diag_embed(s)
        rank = torch.sum(s > 1e-3)
        w = s_diag @ v

        # Acquiring the low rank weight matrix
        w_weight_matrix = w[0:rank, :]
        u_weight_matrix = u[:, 0:rank]

        # Initializing the LS module
        LRS_module = LSModule(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            sparse_matrix=self.sparse_matrix,
            sparsity=self.sparsity,
            degree=self.degree,
        )

        # Assigning the low rank weight matrices (W and U) and the sparse matrix (S) to the LS module
        LRS_module.W_layer.weight = nn.Parameter(w_weight_matrix)
        LRS_module.U_layer.weight = nn.Parameter(u_weight_matrix)
        
        if self.sparse_matrix:
            LRS_module.S_layer.weight = nn.Parameter(s_weight_matrix)
            # Pruning the sparse matrix so that it is not updated during training
            torch.nn.utils.prune.custom_from_mask(
                LRS_module.S_layer, name="weight", mask=(s_weight_matrix != 0) * 1
            )

        return LRS_module

    def initialize_low_rank(self):
        for module_name, module in self.model.hidden_layers.named_modules():
            if isinstance(module, nn.Linear):
                self.model.hidden_layers._modules[module_name] = self._low_rank_sparse(
                    module
                )

        return self.model
