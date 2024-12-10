import torch
import numpy as np

# One-Hot to denote String
# torch.FloatTensor torch.cuda.FloatTensor
# torch.DoubleTensor torch.cuda.DoubleTensor
# torch.IntTensor torch.cuda.IntTensor
# torch.ByteTensor torch.cuda.ByteTensor

a=torch.zeros(3, 4)
print('torch.zeros(3, 4):')
print(a)
print(a.type(), a.shape, a.shape[0], a.shape[1], a.size(), a.size(0), a.size(1), a.dim(), len(a.shape), a.numel())
print(isinstance(a, torch.FloatTensor))

b=torch.ones(3, 4)
print('torch.ones(3, 4):')
print(b)
print(b.type(), b.shape, b.shape[0], b.shape[1], b.size(), b.size(0), b.size(1), b.dim(), len(b.shape), b.numel())
print(isinstance(b, torch.FloatTensor))

c=torch.rand(3, 4)
print('torch.rand(3, 4):')
print(c)
print(c.type(), c.shape, c.shape[0], c.shape[1], c.size(), c.size(0), c.size(1), c.dim(), len(c.shape), c.numel())
print(isinstance(c, torch.FloatTensor))

d=torch.randn(3, 4)
print('torch.randn(3, 4):')
print(d)
print(d.type(), d.shape, d.shape[0], d.shape[1], d.size(), d.size(0), d.size(1), d.dim(), len(d.shape), d.numel())
print(isinstance(d, torch.FloatTensor))

e=torch.full([3, 4], 7)
print('torch.full([3, 4], 7):')
print(e)
print(e.type(), e.shape, e.shape[0], e.shape[1], e.size(), e.size(0), e.size(1), e.dim(), len(e.shape), e.numel())
print(isinstance(e, torch.FloatTensor))

f=torch.normal(mean=torch.ones(3, 4), std=torch.full([3, 4], 0.1))
print('torch.normal(mean=torch.ones(3, 4), std=torch.full([3, 4], 0.1)):')
print(f)
print(f.type(), f.shape, f.shape[0], f.shape[1], f.size(), f.size(0), f.size(1), f.dim(), len(f.shape), f.numel())
print(isinstance(f, torch.FloatTensor))

g=torch.eye(3, 4)
print('torch.eye(3, 4):')
print(g)
print(g.type(), g.shape, g.shape[0], g.shape[1], g.size(), g.size(0), g.size(1), g.dim(), len(g.shape), g.numel())
print(isinstance(g, torch.FloatTensor))

h=torch.tensor([3, 4])
print('torch.tensor([3, 4]):')
print(h)
print(h.type(), h.shape, h.shape[0], h.size(), h.size(0), h.dim(), len(h.shape), h.numel())

i=torch.FloatTensor(3, 4)
print('torch.FloatTensor(3, 4):')
print(i)
print(i.type(), i.shape, i.shape[0], i.shape[1], i.size(), i.size(0), i.size(1), i.dim(), len(i.shape), i.numel())

j=torch.FloatTensor([3, 4])
print('torch.FloatTensor([3, 4]):')
print(j)
print(j.type(), j.shape, j.shape[0],  j.size(), j.size(0),  j.dim(), len(j.shape), j.numel())

k=torch.empty(3, 4)
print('torch.empty(3, 4):')
print(k)
print(k.type(), k.shape, k.shape[0], k.shape[1], k.size(), k.size(0), k.size(1), k.dim(), len(k.shape), k.numel())

torch.set_default_tensor_type(torch.DoubleTensor)
print('After torch.set_default_tensor_type(torch.DoubleTensor):')

l=torch.from_numpy(np.ones(3))
print('torch.from_numpy(np.ones(3)):')
print(np.ones(3))
print(l)
print(l.type(), l.shape, l.shape[0], l.size(), l.size(0), l.dim(), len(l.shape), l.numel())

m=torch.zeros_like(a)
print('torch.zeros_like(a):')
print(m)
print(m.type(), m.shape, m.shape[0], m.shape[1], m.size(), m.size(0), m.size(1), m.dim(), len(m.shape), m.numel())

n=torch.ones_like(e)
print('torch.ones_like(e):')
print(n)
print(n.type(), n.shape, n.shape[0], n.shape[1], n.size(), n.size(0), n.size(1), n.dim(), len(n.shape), n.numel())

o=torch.full_like(a, 8)
print('torch.full_like(a, 8):')
print(o)
print(o.type(), o.shape, o.shape[0], o.shape[1], o.size(), o.size(0), o.size(1), o.dim(), len(o.shape), o.numel())

p=torch.arange(0, 10)
print('torch.arange(0, 10):')
print(p)
print(p.type(), p.shape, p.shape[0], p.size(), p.size(0), p.dim(), len(p.shape), p.numel())

q=torch.arange(0, 10, 2)
print('torch.arange(0, 10, 2):')
print(q)
print(q.type(), q.shape, q.shape[0], q.size(), q.size(0), q.dim(), len(q.shape), q.numel())

r=torch.linspace(0, 10, 4)
print('torch.linspace(0, 10, 4):')
print(r)
print(r.type(), r.shape, r.shape[0], r.size(), r.size(0), r.dim(), len(r.shape), r.numel())

s=torch.logspace(0, 10, 5)
print('torch.logspace(0, 10, 5):')
print(s)
print(s.type(), s.shape, s.shape[0], s.size(), s.size(0), s.dim(), len(s.shape), s.numel())

t=torch.randperm(3)
print(t.type(), t.shape, t.shape[0], t.size(), t.size(0), t.dim(), len(t.shape), t.numel())
print('torch.randperm(3):')
print(t)
print('g[t]:')
print(g[t])
t=torch.randperm(3)
print('torch.randperm(3):')
print(t)
print('g[t]:')
print(g[t])

u=torch.full([], 7)
print('torch.full([], 7):')
print(u)
print(u.type(), u.shape, u.size(), u.dim(), len(u.shape), u.numel())

v=torch.rand_like(a)
print('torch.rand_like(a):')
print(v)
print(v.type(), v.shape, v.shape[0], v.shape[1], v.size(), v.size(0), v.size(1), v.dim(), len(v.shape), v.numel())

w=torch.randn_like(a)
print('torch.randn_like(a):')
print(w)
print(w.type(), w.shape, w.shape[0], w.shape[1], w.size(), w.size(0), w.size(1), w.dim(), len(w.shape), w.numel())