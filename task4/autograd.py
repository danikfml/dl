import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpCosFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        result = torch.exp(x) + torch.cos(y)
        ctx.save_for_backward(x, y)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = torch.exp(x) * grad_output
        grad_y = -torch.sin(y) * grad_output
        return grad_x, grad_y


x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=True)

result = ExpCosFunction.apply(x, y)
result.sum().backward()
print(x.grad, y.grad)

result_standard = torch.exp(x) + torch.cos(y)
result_standard.sum().backward()
print(torch.allclose(x.grad, x.grad), torch.allclose(y.grad, y.grad))
