import torch


condition = torch.rand(2)
print(condition < 0.5)
x = torch.rand(2, 3, 3)
y = torch.rand(2, 3, 3)
print(x)
print(y)
print(torch.where((condition < 0.5).reshape(2, 1, 1), x, y))
