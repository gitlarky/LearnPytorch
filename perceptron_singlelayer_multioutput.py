import torch

x=torch.randn(1, 10)
print('x=torch.randn(1, 10): x.size()=', x.size(), 'x= \n', x)
w=torch.randn(2, 10, requires_grad=True)
print('w=torch.randn(2, 10, requires_grad=True): w.size()=', w.size(), 'w= \n', w)
theta=torch.sigmoid(x@w.t())
print('theta=torch.sigmoid(x@w.t()): theta.size()=', theta.size(), 'theta= \n', theta)
e=torch.nn.functional.mse_loss(torch.ones(theta.size()), theta)
print('e=torch.nn.functional.mse_loss(torch.ones(theta.size()), theta): e.size()=', e.size(), 'e= \n', e)
w_grad=torch.autograd.grad(e, w)
print('w_grad=torch.autograd.grad(e, w): w_grad= \n', w_grad[0])

w_grad_manual=((theta-1)*theta*(1-theta)).t()@x
print('w_grad_manual=((theta-1)*theta*(1-theta)).t()@x: w_grad_manual.size()=', w_grad_manual.size(), 'w_grad_manual= \n', w_grad_manual)

print('w_grad==w_grad_manual: ', torch.equal(w_grad[0], w_grad_manual))