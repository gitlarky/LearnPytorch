import torch

x=torch.randn(1, 10)
print('x=torch.randn(1, 10): x.size()=', x.size(), 'x= \n', x)
w=torch.randn(1, 10, requires_grad=True)
print('w=torch.randn(1, 10, requires_grad=True): w.size()=', w.size(), 'w= \n', w)
theta=torch.sigmoid(x@w.t())
print('theta=torch.sigmoid(x@w.t()): theta.size()=', theta.size(), 'theta= \n', theta)
e=torch.nn.functional.mse_loss(torch.ones(theta.size()), theta)
print('e=torch.nn.functional.mse_loss(torch.ones(theta.size()), theta): e.size()=', e.size(), 'e= \n', e)
w_grad=torch.autograd.grad(e, w)
print('w_grad=torch.autograd.grad(e, w): w_grad= \n', w_grad)