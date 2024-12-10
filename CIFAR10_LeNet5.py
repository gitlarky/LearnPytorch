# To have attractive lips, speak kind words.
# To have a loving look, look for the good side of people.
#================================================ import module ===================================
import torch
import torchvision

from visdom import Visdom
from sklearn.model_selection import KFold

#================================================ set parameters ==================================
maxepoch=20
k_folds=5
batch_size=100
learning_rate=1e-2
lan_l1=0
lan_l2=0.01
momentum=0.9
channel_size=3
h_image=32
w_image=32
device=torch.device('cuda:0')

#================================================ get data ========================================
train_data=torchvision.datasets.CIFAR10('../../../data', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.Resize((h_image, w_image)),
                       torchvision.transforms.ToTensor()
                   ]))

test_data=torchvision.datasets.CIFAR10('../../../data', train=False, 
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.Resize((h_image, w_image)),
                       torchvision.transforms.ToTensor()
                   ]))
print('Train Data Size:', len(train_data), '; Test Data Size:', len(test_data))

#================================================ define neural network ===========================
class myLeNet5(torch.nn.Module): # for CIFAR10 image classification

    def __init__(self):
        super(myLeNet5, self).__init__()

        self.model=torch.nn.Sequential(
            torch.nn.Conv2d(channel_size, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(16*5*5, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )

        # self.conv_unit=torch.nn.Sequential(
            # torch.nn.Conv2d(channel_size, 6, kernel_size=5, stride=1, padding=0),
            # torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        # )

        # self.fc_unit=torch.nn.Sequential(
            # torch.nn.Linear(16*5*5, 120),
            # torch.nn.ReLU(),
            # torch.nn.Linear(120, 84),
            # torch.nn.ReLU(),
            # torch.nn.Linear(84, 10)
        # )

    
    def forward(self, x):
        # bsz=x.size(0)
        # x  =self.conv_unit(x)
        # x  =x.view(bsz, 16*5*5)
        # x  =self.fc_unit(x)
        x=self.model(x)

        return x

#================================================ train and validate neural network ===============
myNet=myLeNet5().to(device)
# print('Neural Network Parameters Size:', len(myNet.parameters()))

optimizer=torch.optim.SGD(myNet.parameters(), lr=learning_rate, weight_decay=lan_l2, momentum=momentum)
loss_function=torch.nn.CrossEntropyLoss().to(device)

scheduler1=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2, verbose=True)
scheduler2=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

viz=Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='Train Loss'))
viz.line([0.], [0.], win='val', opts=dict(title='Validation Accuracy'))
global_step=0

kfold = KFold(n_splits=k_folds, shuffle=True)
train_ids_set=[]
val_ids_set=[]
for t, v in kfold.split(train_data):
    train_ids_set.append(t)
    val_ids_set.append(v)


for epoch in range(maxepoch):

    train_ids=train_ids_set[epoch%k_folds]
    val_ids  =  val_ids_set[epoch%k_folds]

    train_subsampler=torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler  =torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_subsampler)
    val_loader  =torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=  val_subsampler)

    print('Train: {} batches; Validate: {} batches; (Batch Size: {})'.format(len(train_loader), len(val_loader), batch_size))

    myNet.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.to(device), target.to(device)
        logits=myNet(data)
        loss=loss_function(logits, target)

        loss_l1=0
        for parm in myNet.parameters():
            loss_l1+=torch.sum(torch.abs(parm))
        loss+=lan_l1*loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1)%50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(epoch, (batch_idx+1)*len(data), batch_size*len(train_loader), 100*batch_idx/len(train_loader), loss.item()/batch_size))
        
        global_step+=1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

    myNet.eval()    
    val_loss=0
    correct=0
    for data, target in val_loader:
        data, target=data.to(device), target.to(device)
        logits=myNet(data)
        val_loss+=loss_function(logits, target).item()

        pred=logits.data.argmax(dim=1)
        correct+=pred.eq(target.data).sum()

        viz.images(data.view(-1, 1, h_image, w_image).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
        viz.text(str(pred), win='pred', opts=dict(title='Predicted'))

    val_loss/=(batch_size*len(val_loader))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, (batch_size*len(val_loader)), 100*correct/(batch_size*len(val_loader))))

    viz.line([(correct/(batch_size*len(val_loader))).cpu().numpy()], [epoch], win='val', update='append')

    scheduler1.step()
    scheduler2.step(val_loss)

#================================================ test neural network =============================
myNet.eval()  
test_loader=torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_loss=0
correct=0
for data, target in test_loader:
    data, target=data.to(device), target.to(device)
    logits=myNet(data)
    test_loss+=loss_function(logits, target).item()

    pred=logits.data.argmax(dim=1)
    correct+=pred.eq(target.data).sum()

    viz.images(data.view(-1, 1, h_image, w_image).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
    viz.text(str(pred), win='pred', opts=dict(title='Predicted'))

test_loss/=len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))





