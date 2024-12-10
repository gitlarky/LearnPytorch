import torch
import torchvision

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

layer1=torch.nn.Linear(784, 200)
layer2=torch.nn.Linear(200, 200)
layer3=torch.nn.Linear(200, 10)

def neunet(x):
    x=layer1(x)
    x=torch.nn.functional.relu(x, inplace=True)
    x=layer2(x)
    x=torch.nn.functional.relu(x, inplace=True)
    x=layer3(x)
    x=torch.nn.functional.relu(x, inplace=True)

    return x

optimizer=torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
loss_function=torch.nn.CrossEntropyLoss()

for epoch in range(maxepoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        data=data.view(-1, 28*28)
        logits=neunet(data)
        loss=loss_function(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.item()))
    
    test_loss=0
    correct=0
    for data, target in test_loader:
        data=data.view(-1, 28*28)
        logits=neunet(data)
        test_loss+=loss_function(logits, target).item()

        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()

    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))



