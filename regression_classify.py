import torch

a=torch.linspace(-10, 10, 11)
print('a: \n', a)
asigmoid=torch.sigmoid(a)
print('asigmoid=torch.sigmoid(a): \n', asigmoid)
atanh   =torch.tanh(a)
print('atanh   =torch.tanh(a):    \n', atanh)
arelu   =torch.relu(a)
print('arelu   =torch.relu(a):    \n', arelu)

sample_count=100
xmin=5
xmax=10
w=torch.tensor([2.])
b=torch.tensor([3.])
epsilon=0.1
print('The original function is: y=', w, 'x+', b)
X=torch.linspace(xmin, xmax, sample_count)
print('X.size()=', X.size(), 'X: \n', X)
Y=torch.zeros(sample_count)
E=torch.normal(mean=torch.zeros(sample_count), std=torch.full([sample_count], epsilon))
print('E.size()=', E.size(), 'E: \n', E)
Y+=w*X+b+E
print('Y.size()=', Y.size(), 'Y: \n', Y)
w.requires_grad_()
mse_auto=torch.nn.functional.mse_loss(Y, w*X+b, size_average=True)
print('mse_auto  =torch.nn.functional.mse_loss(Y, w*X+b, size_average=True): ', mse_auto)
mse_manual=torch.sum((Y-w*X-b)**2)/sample_count
print('mse_manual=mse_manual=torch.sum((Y-w*X-b)**2)/sample_count          : ', mse_auto)
w_grad_auto  =torch.autograd.grad(mse_auto, w)
print('w_grad_auto  =torch.autograd.grad(mse_auto, w)          :', w_grad_auto)
w_grad_manual=torch.sum(2*(Y-(w*X+b))*(-X))/sample_count
print('w_grad_manual=torch.sum(2*(Y-(w*X+b))*(-X))/sample_count:', w_grad_manual)

mse_auto=torch.nn.functional.mse_loss(Y, w*X+b, size_average=True)
mse_auto.backward()
print('mse_auto.backward(), w.grad                             :', w.grad)

w_grad_half  =torch.autograd.grad(mse_manual, w)
print('w_grad_half  =torch.autograd.grad(mse_manual, w)        :', w_grad_half)

YY=torch.sum(w*X+b+E)/sample_count
w_YY_grad=torch.autograd.grad(YY, w)
print('w_YY_grad=torch.autograd.grad(YY, w):', w_YY_grad)



s=torch.rand(3)
s.requires_grad_()
p=torch.nn.functional.softmax(s, dim=0)
print('s=torch.rand(4):', s)
print('p=torch.nn.functional.softmax(s, dim=0):', p)
print('torch.sum(p):', torch.sum(p))
# print('torch.autograd.grad(p[0], s, retain_graph=True): \n', torch.autograd.grad(p[0], s, retain_graph=True))
print('torch.autograd.grad(p[1], s): \n', torch.autograd.grad(p[1], s))
# print('torch.autograd.grad(p[2], s): \n', torch.autograd.grad(p[2], s))