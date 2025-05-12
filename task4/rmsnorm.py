import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


input_tensor = torch.randn(10, 20)
rmsnorm_layer = RMSNorm(20)
output = rmsnorm_layer(input_tensor)

torch_rmsnorm = nn.RMSNorm(20)
torch_output = torch_rmsnorm(input_tensor)
print(torch.allclose(output, torch_output))
