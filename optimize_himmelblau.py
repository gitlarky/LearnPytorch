import torch
import numpy
import matplotlib.pyplot as plt

samplecount=100
maxstep=20000
learningrate=1e-3

x=torch.linspace(-6, 6, samplecount)
y=torch.linspace(-6, 6, samplecount)
print('x.size()=', x.size(), '; y.size()=', y.size())

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

X, Y=numpy.meshgrid(x, y)
print('X.shape=', X.shape, '; Y.shape=', Y.shape)
Z=himmelblau([X, Y])
print('Z.shape=', Z.shape)

fig=plt.figure('Himmelblau')
ax=plt.subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

p=torch.tensor([0., 0.], requires_grad=True)
optimizer=torch.optim.Adam([p], lr=learningrate)
print('p.shape=', p.shape, '\np:', p, '\n[p]:', [p])

for i in range(maxstep):
    pred=himmelblau(p)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if i%2000==0:
        print('#{}: p={}, himmelblau(x)={}'.format(i, p.tolist(), pred.item()))

