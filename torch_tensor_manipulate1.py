import torch

a=torch.FloatTensor(2, 3, 4, 4)
print(a[0].shape, a[0,0].shape, a[0,0,0].shape)
print(a[0].size(), a[0,0].size(), a[0,0,0].size())

print('a=torch.FloatTensor(2, 3, 4, 4):')
print(a)
print('a[1:2, :1, ..., ::2]:')
print(a[1:2, :1, ..., ::2])
print('a[-1:, :1, ..., ::2]:')
print(a[-1:, :1, ..., ::2])
print('a[-1:, :1, ::, ::2]:')
print(a[-1:, :1, ::, ::2])
print('a[..., :2, ...]:')
print(a[..., :2, ...])
print('a[..., :2]:')
print(a[..., :2])
print('a[..., ::2]:')
print(a[..., ::2])

print('a.index_select(0, torch.tensor([1])):')
print(a.index_select(0, torch.tensor([1])))
print('a.index_select(1, torch.arange(2)):')
print(a.index_select(1, torch.arange(2)))
print('a.index_select(3, torch.tensor([0, 2])):')
print(a.index_select(3, torch.tensor([0, 2])))

b=torch.randn(3, 4)
print('torch.randn(3, 4):')
print(b)
print('b.masked_select(b.le(0)):')
print(b.masked_select(b.le(0)))
mask=torch.randn(3, 4).ge(0)
print(mask)
print('b.masked_select(mask); mask=b.ge(0):')
print(b.masked_select(mask))
cond=torch.randn(3, 4)
print('cond=torch.randn(3, 4)\n', cond)
print('torch.where(cond>=0, b, cond):\n', torch.where(cond>=0, b, cond))

c=torch.tensor([[1, 2, 3], [4, 5, 6]])
print('torch.tensor([[1, 2, 3], [4, 5, 6]]):')
print(c)
print('torch.take(c, torch.tensor([0, 2, 5])):')
print(torch.take(c, torch.tensor([0, 2, 5])))

print('a=torch.FloatTensor(2, 3, 4, 4):')
print(a)
d=a.view(2*3*4*4)
print('d=a.view(2*3*4*4):')
print(d)
print('e=d.view(2, 3*4, 4):')
e=d.view(2, 3*4, 4)
print(e)
f=e.reshape(2, 3, 4, 4)
print('f=e.reshape(2, 3, 4, 4):')
print(f)
print('f==a:', f==a)

g=torch.tensor([1., 2.])
print('g:\n', g, '\n g.size()=', g.size())
h=g.unsqueeze(0)
print('h=g.unsqueeze(0):\n', h, '\n h.size()=', h.size())
print('g:\n', h)
i=h.unsqueeze(2)
print('i=h.unsqueeze(2):\n', i, '\n i.size()=', i.size())
j=i.unsqueeze(3)
print('j=i.unsqueeze(3):\n', j, '\n j.size()=', j.size())
print('g.unsqueeze(-2).unsqueeze(-1).unsqueeze(-1).size()=', g.unsqueeze(-2).unsqueeze(-1).unsqueeze(-1).size())

m=j.squeeze(0)
print('m=j.squeeze(0); m.size()=', m.size())
n=j.squeeze()
print('n=m.squeeze(); n.size()=', n.size(), 'n==g: ', n==g)

k=j.expand(2, 2, 1, 1)
print('k=j.expand(2, 2, 1, 1): \n', k, '\n k.size()=', k.size())
k=j.expand(2, -1, 1, 1)
print('k=j.expand(2, -1, 1, 1): \n', k, '\n k.size()=', k.size())
l=k.repeat(1, 1, 3, 3)
print('l=k.repeat(1, 1, 3, 3): \n', l, '\n l.size()=', l.size())

o=torch.randn(3, 4)
print('o.size()=', o.size(), 'o: \n', o)
print('o.t().size()=', o.t().size(), 'o.t(): \n', o.t())

p=torch.randn(2, 3, 4, 5)
q=p.transpose(1, 3).transpose(1, 2)
print('q.size():', q.size())
r=p.permute(0, 2, 3, 1)
print('r.size():', r.size())
print('torch.all(torch.eq(r, q)):', torch.all(torch.eq(r, q)))
s=q.view(2, 4*5, 3)
print('s=q.view(2, 4*5, 3):', s.size())
t=q.contiguous().view(2*4*5*3)
print('t=q.contiguous().view(2*4*5*3):', t.size())
u=q.reshape(2, 4*5*3)
print('t=q.reshape(2, 4*5*3):', t.size())
v=t.view(2, 4, 5, 3)
w=u.view(2, 4, 5, 3)
print(torch.all(torch.eq(v, w)))
# t=s.contiguous()
# print(torch.all(torch.eq(s, t)))