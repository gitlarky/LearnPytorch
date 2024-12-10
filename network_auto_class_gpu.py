import torch
import torchvision
from visdom import Visdom

learning_rate=1e-2
maxepoch=10
batch_size=200
device=torch.device('cuda:0')

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


class myMLP(torch.nn.Module):

    def __init__(self):
        super(myMLP, self).__init__()

        self.model=torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(200, 10),
        )
    
    def forward(self, x):
        x=self.model(x)

        return x

myNet=myMLP().to(device)

optimizer=torch.optim.SGD(myNet.parameters(), lr=learning_rate)
loss_function=torch.nn.CrossEntropyLoss().to(device)

viz=Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='Train Loss'))
viz.line([0.], [0.], win='test', opts=dict(title='Test Accuracy'))
global_step=0

for epoch in range(maxepoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        data=data.view(-1, 28*28)
        data, target=data.to(device), target.to(device)
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
        data, target=data.to(device), target.to(device)
        logits=myNet(data)
        test_loss+=loss_function(logits, target).item()

        # pred=logits.data.max(1)[1]
        pred=logits.data.argmax(dim=1)
        correct+=pred.eq(target.data).sum()

        viz.images(data.view(-1, 1, 28, 28).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
        viz.text(str(pred), win='pred', opts=dict(title='Predicted'))

    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

    viz.line([(correct/len(test_loader.dataset)).cpu().numpy()], [epoch], win='test', update='append')




