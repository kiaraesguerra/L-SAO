import torch
import torch.nn as nn
from ls_initializers.ramanujan_constructions import Ramanujan_Constructions
from initializers.weight_init import _ortho_generator


class LSModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        sparse_matrix: str = None,
        sparsity: float = None,
        degree: float = None,
        layer_type: str = "linear",
    ):
        super(LSModule, self).__init__()

        self.rank = rank
        self.sparse_matrix = sparse_matrix
        self.degree = degree
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = layer_type

        if self.layer_type == "linear":
            self.W_layer = nn.Linear(
                in_features=in_features, out_features=rank, bias=False
            )
            self.U_layer = nn.Linear(
                in_features=rank, out_features=out_features, bias=False
            )
        else:
            self.W_layer = nn.Conv1d(
                in_channels=in_features, out_channels=rank, kernel_size=1, bias=False
            )
            self.U_layer = nn.Conv1d(
                in_channels=rank, out_channels=out_features, kernel_size=1, bias=False
            )

        if self.sparse_matrix:
            if self.layer_type == "linear":
                self.S_layer = nn.Linear(
                    in_features=in_features, out_features=out_features, bias=False
                )
            else:
                self.S_layer = nn.Conv1d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=1,
                    bias=False,
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
        threshold: float = 1e-3,
        sparsity: float = None,
        degree: int = None,
        activation: str = "tanh",
        rank: int = None,
    ):
        self.sparse_matrix = sparse_matrix
        self.threshold = threshold
        self.model = model
        self.sparsity = sparsity
        self.degree = degree
        self.activation = activation
        self.rank = rank

    def _sparse_matrix(self, module):
        if self.sparse_matrix in ["SAO", "RG-N", "RG-U"]:
            constructor = Ramanujan_Constructions(
                height=module.weight.shape[0],
                width=module.weight.shape[1],
                method=self.sparse_matrix,
                sparsity=self.sparsity,
                degree=self.degree,
                activation=self.activation,
            )
            s_weight_matrix, _ = constructor()

        elif self.sparse_matrix == "LMP":
            s_module = nn.Linear(module.out_features, module.in_features).to("cuda")
            s_module.weight = nn.Parameter(_ortho_generator(module, self.activation))
            torch.nn.utils.prune.ln_structured(
                s_module, name="weight", amount=self.sparsity, n=2, dim=0
            )
            s_weight_matrix = s_module.weight

        return s_weight_matrix

    def _low_rank_sparse(self, module, layer_type):
        LR = module.weight.reshape(module.weight.shape[0], -1).to("cuda")

        if self.sparse_matrix:
            s_weight_matrix = self._sparse_matrix(module)
            LR = LR - s_weight_matrix

        if not self.rank:
            # Performing SVD on the difference matrix
            padded_s = torch.zeros(module.weight.shape[0], module.weight.shape[1]).to(
                "cuda"
            )
            u, s, v = torch.linalg.svd(LR)
            s_diag = torch.diag_embed(s)
            padded_s[0 : s_diag.shape[0], 0 : s_diag.shape[1]] = s_diag
            rank = torch.sum(s > self.threshold)
            w = padded_s @ v
            w_weight_matrix = w[0:rank, :]
            u_weight_matrix = u[:, 0:rank]
        else:
            rank = self.rank

        LRS_module = LSModule(
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
            rank=rank,
            sparse_matrix=self.sparse_matrix,
            sparsity=self.sparsity,
            degree=self.degree,
            layer_type=layer_type,
        )

        if self.rank:
            LRS_module.W_layer.weight = nn.Parameter(
                _ortho_generator(LRS_module.W_layer, self.activation).reshape(
                    LRS_module.W_layer.weight.shape
                )
            )
            LRS_module.U_layer.weight = nn.Parameter(
                _ortho_generator(LRS_module.U_layer, self.activation).reshape(
                    LRS_module.U_layer.weight.shape
                )
            )
        else:
            LRS_module.W_layer.weight = nn.Parameter(
                w_weight_matrix.reshape(LRS_module.W_layer.weight.shape)
            )
            LRS_module.U_layer.weight = nn.Parameter(
                u_weight_matrix.reshape(LRS_module.U_layer.weight.shape)
            )

        if self.sparse_matrix:
            LRS_module.S_layer.weight = nn.Parameter(s_weight_matrix)
            torch.nn.utils.prune.custom_from_mask(
                LRS_module.S_layer, name="weight", mask=(s_weight_matrix != 0) * 1
            )

        return LRS_module

    def initialize_low_rank_mlp(self):
        for module_name, module in self.model.hidden_layers.named_modules():
            if isinstance(module, nn.Linear):
                self.model.hidden_layers._modules[module_name] = self._low_rank_sparse(
                    module
                )

    def initialize_low_rank_mixer(self):
        for module_name, module in self.model.mixer_layers.named_modules():
            if isinstance(module, nn.Linear):
                name = list(module_name.split("."))
                setattr(
                    self.model.mixer_layers[int(name[0])].mlp2,
                    name[2],
                    self._low_rank_sparse(module, layer_type="linear"),
                )

            elif isinstance(module, nn.Conv1d):
                name = list(module_name.split("."))
                setattr(
                    self.model.mixer_layers[int(name[0])].mlp1,
                    name[2],
                    self._low_rank_sparse(module, layer_type="conv1d"),
                )

        return self.model
