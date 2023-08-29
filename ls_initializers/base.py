import torch
import torch.nn as nn


class Base:
    def __init__(
        self,
        module: nn.Module,
        activation: str = "relu",
        device: str = "cuda",
    ):
        self.in_ = module.weight.shape[1]
        self.out_ = module.weight.shape[0]
        self.rows = min(self.out_, self.in_)
        self.columns = max(self.out_, self.out_)
        self.device = device
        self.activation = activation
        self.ramanujan_mask = None

    def _ortho_gen(self, rows, columns) -> torch.tensor:
        rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
        q, _ = torch.qr(rand_matrix)
        orthogonal_matrix = q[:, :columns]
        return orthogonal_matrix.T if columns > rows else orthogonal_matrix

    def _ortho_generator(self) -> torch.tensor:
        if self.activation == "relu" and self.in_ch != 3:
            rows = self.out_ch // 2
            columns = self.in_ch // 2
            orthogonal_matrix = self._concat(self._ortho_gen(rows, columns))

        else:
            rows = self.out_ch
            columns = self.in_ch
            orthogonal_matrix = self._ortho_gen(rows, columns)

        return orthogonal_matrix