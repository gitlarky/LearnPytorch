import torch

# ====================== input ======================================
sample_count=100
xmin=5
xmax=10
w=2
b=3
epsilon=0.1
current_w=1
current_b=2
learning_rate=0.01
tolerance=1e-10
max_iteration=50000

# ============================== define points =========================
print('The original function is: y=', w, 'x+', b)
X=torch.linspace(xmin, xmax, sample_count)
print('X.size()=', X.size(), 'X: \n', X)
Y=torch.zeros(sample_count)
E=torch.normal(mean=torch.zeros(sample_count), std=torch.full([sample_count], epsilon))
print('E.size()=', E.size(), 'E: \n', E)
Y+=w*X+b+E
print('Y.size()=', Y.size(), 'Y: \n', Y)
Points=torch.cat([X.view(sample_count, 1), Y.view(sample_count, 1)], dim=1)
print('Points.size()=', Points.size(), 'Points: \n', Points)

# ====================== calculate gradient ==========================
def gradient_w(p, w0, b0):
    g_w=0
    for i in range(len(p)):
        g_w+=2*(w0*p[i][0]+b0-p[i][1])*p[i][0]
    return g_w/len(p)
def gradient_b(p, w0, b0):
    g_b=0
    for i in range(len(p)):
        g_b+=2*(w0*p[i][0]+b0-p[i][1])
    return g_b/len(p)
# ============================ iterating solve w and b ====================
it=1
max_relative_error=1
while (it<=max_iteration) & (max_relative_error>tolerance):
    new_w=current_w-learning_rate*gradient_w(Points, current_w, current_b)
    new_b=current_b-learning_rate*gradient_b(Points, current_w, current_b)
    max_relative_error=max(abs((new_b-current_b)/new_b), abs((new_w-current_w)/new_w))
    print('#', it, ': w=', new_w, 'b=', new_b, 'err=', max_relative_error)
    it+=1
    current_w=new_w
    current_b=new_b

