import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LionOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, beta1=0.95, beta2=0.98, weight_decay=0.1):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                lr = group['lr']
                beta1 = group['beta1']
                beta2 = group['beta2']
                wd = group['weight_decay']
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)

                momentum = state['momentum']
                update = (1 - beta1) * grad + beta1 * momentum
                update = torch.sign(update)
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)


class EnhancedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.LayerNorm(256),
            Swish(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            Swish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


def generate_data(num_samples=2048):
    X = torch.randn(num_samples, 10)
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    w_true = torch.randn(10, 2) * 2.5
    b_true = torch.randn(2)
    y_linear = X @ w_true + b_true
    y_nonlinear = torch.sin(X[:, :3]).sum(dim=1, keepdim=True) * 0.7
    y = y_linear + y_nonlinear
    y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-6)
    y += 0.005 * torch.randn_like(y)
    return X, y


X, y = generate_data()
model = EnhancedModel()
optimizer = LionOptimizer(model.parameters(), lr=3e-3, weight_decay=0.1)
criterion = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            preds = model(X)
            mse = torch.mean((preds - y) ** 2)
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | MSE: {mse.item():.4f}')

with torch.no_grad():
    preds = model(X)
    final_mse = torch.mean((preds - y) ** 2).item()
    print(f"\nFinal MSE: {final_mse:.4f}")
    print(f"Predictions std: {preds.std().item():.2f} | Targets std: {y.std().item():.2f}")
    print(f"Max error: {torch.abs(preds - y).max().item():.2f}")
