import torch
import torch.nn as nn
from torch import Tensor


class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class LoanApprovalModel(nn.Module):
    def __init__(self, input_numeric_dim: int, num_cat: dict[str, int], hidden_size: int,
                 n_blocks: int = 1, use_skip: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_classes, hidden_size) for feat, num_classes in num_cat.items()
        })
        self.numeric_linear = nn.Linear(input_numeric_dim, hidden_size)
        self.input_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.blocks = nn.ModuleList([BaseBlock(hidden_size) for _ in range(n_blocks)])
        self.use_skip = use_skip
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        x_numeric = torch.stack(list(numeric_features.values()), dim=-1)
        x_numeric = self.numeric_linear(x_numeric)
        emb_list = [emb_layer(cat_features[feat]) for feat, emb_layer in self.embeddings.items()]
        x_cat = torch.cat([emb.unsqueeze(1) for emb in emb_list], dim=1)
        x_cat = torch.sum(x_cat, dim=1)
        x = torch.cat([x_numeric, x_cat], dim=-1)
        x = self.input_linear(x)
        x = self.dropout(x)
        for block in self.blocks:
            out = block(x)
            x = x + out if self.use_skip else out
            x = self.dropout(x)
        result = self.output_layer(x)
        result = result.squeeze(-1)
        return result
