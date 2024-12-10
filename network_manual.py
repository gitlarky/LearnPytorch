import torch
import torchvision
from visdom import Visdom

learning_rate=1e-2
maxepoch=10
batch_size=200

train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../../../data', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../../../data', train=False, 
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

w1, b1=torch.randn(200, 784, requires_grad=True), torch.randn(200, requires_grad=True)
w2, b2=torch.randn(200, 200, requires_grad=True), torch.randn(200, requires_grad=True)
w3, b3=torch.randn(10 , 200, requires_grad=True), torch.randn(10 , requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def myNet(x):
    x=x@w1.t()+b1
    x=torch.nn.functional.relu(x)
    x=x@w2.t()+b2
    x=torch.nn.functional.relu(x)
    x=x@w3.t()+b3
    # x=torch.nn.functional.relu(x)

    return x

optimizer=torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
loss_function=torch.nn.CrossEntropyLoss()

viz=Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='Train Loss'))
viz.line([0.], [0.], win='test', opts=dict(title='Test Accuracy'))
global_step=0

for epoch in range(maxepoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        data=data.view(-1, 28*28)
        logits=myNet(data)
        loss=loss_function(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.item()))

        global_step+=1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

    test_loss=0
    correct=0
    for data, target in test_loader:
        data=data.view(-1, 28*28)
        logits=myNet(data)
        test_loss+=loss_function(logits, target).item()

        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()

        viz.images(data.view(-1, 1, 28, 28).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
        viz.text(str(pred), win='pred', opts=dict(title='Predicted'))

    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

    viz.line([correct/len(test_loader.dataset)], [epoch], win='test', update='append')
