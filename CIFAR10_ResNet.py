# To have attractive lips, speak kind words.
# To have a loving look, look for the good side of people.
#================================================ import module ===================================
import torch
import torchvision

from visdom import Visdom
from sklearn.model_selection import KFold

#================================================ set parameters ==================================
maxepoch=5
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
#------------------ ResBlk ------------------------------------------------------------------------
class ResBlk(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1=torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1  =torch.nn.BatchNorm2d(ch_out)
        self.conv2=torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2  =torch.nn.BatchNorm2d(ch_out)

        self.extra=torch.nn.Sequential()
        if ch_out!=ch_in:
            self.extra=torch.nn.Sequential(
                torch.nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                torch.nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out=self.bn1(self.conv1(x))
        out=torch.nn.functional.relu(out)
        out=self.bn2(self.conv2(out))
        out=self.extra(x)+out

        return out
#------------------ ResNet18 ----------------------------------------------------------------------
class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.blk1=ResBlk(64, 64)
        self.blk2=ResBlk(64, 128)
        self.blk3=ResBlk(128, 256)
        self.blk4=ResBlk(256, 512)

        self.outlayer=torch.nn.Linear(512*32*32, 10)

    def forward(self, x):
        x=self.conv1(x)

        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)

        x=x.view(x.size(0), -1)

        x=self.outlayer(x)

        return x
#================================================ train and validate neural network ===============
myNet=ResNet18().to(device)
# print('Neural Network Parameters Size:', len(myNet.parameters()))

optimizer=torch.optim.SGD(myNet.parameters(), lr=learning_rate, weight_decay=lan_l2, momentum=momentum)
loss_function=torch.nn.CrossEntropyLoss().to(device)

scheduler1=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2, verbose=True)
scheduler2=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

viz=Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='Train Loss'))
viz.line([0.], [0.], win='eval', opts=dict(title='Validation Accuracy'))
global_step=0

kfold = KFold(n_splits=k_folds, shuffle=True)
train_ids_set = []
eval_ids_set = []
for t, v in kfold.split(train_data):
    train_ids_set.append(t)
    eval_ids_set.append(v)

#------------------ evaluate ----------------------------------------------------------------------
def evaluate(model, loader, loader_name, device='cpu'):
    model.eval()
    eval_loss, correct = 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        logits       = model(data)
        eval_loss   += loss_function(logits, target).item()

        pred         = logits.data.argmax(dim=1)
        correct     += pred.eq(target.data).sum()

    eval_loss /= len(loader)

    print('\tEvaluation on {}, include {} data:\n\t\tAverage Loss: {:.4e};\n\t\tAccuracy    : {:6.2f}% ({:6d}/{:6d})\n'.format(
        loader_name, len(loader), eval_loss, 100*correct/len(loader), correct, len(loader)))

    return eval_loss, correct
#--------------------------------------------------------------------------------------------------
print('\nStart training...\n')
for epoch in range(maxepoch):

    train_ids = train_ids_set[epoch%k_folds]
    eval_ids  =  eval_ids_set[epoch%k_folds]

    train_subsampler =torch.utils.data.SubsetRandomSampler(train_ids)
    eval_subsampler  =torch.utils.data.SubsetRandomSampler(eval_ids)

    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_subsampler, batch_size=batch_size)
    eval_loader  = torch.utils.data.DataLoader(train_data, sampler= eval_subsampler)

    print('Epoch #{:6d}: Train on {} batches (Batch Size: {})'.format(epoch+1, len(train_loader), batch_size, len(eval_loader)))

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
            print('\tTrained {:3.0f}% ({:6d}/{:6d}),\tAverage Loss: {:.4e}'.format(
                100*batch_idx/len(train_loader), (batch_idx+1)*len(data), batch_size*len(train_loader), loss/batch_size))
        
        global_step+=1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

    eval_loss, correct = evaluate(myNet, eval_loader, 'Validation Data', device)
    viz.images(data.view(-1, 1, h_image, w_image).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
    viz.text(str(pred), win='pred', opts=dict(title='Predicted'))
    viz.line([(correct/len(eval_loader)).cpu().numpy()], [epoch], win='eval', update='append')

    scheduler1.step()
    scheduler2.step(eval_loss)

#================================================ test neural network =============================
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)
test_size   = len(test_loader.dataset)
evaluate(myNet, test_loader, 'Test Data', device)
# test_loss=0
# correct=0
# for data, target in test_loader:
#     data, target=data.to(device), target.to(device)
#     logits=myNet(data)
#     test_loss+=loss_function(logits, target).item()

#     pred=logits.data.argmax(dim=1)
#     correct+=pred.eq(target.data).sum()

    # viz.images(data.view(-1, 1, h_image, w_image).clamp(0, 1), win='pics', opts=dict(title='Handwirtting'))
    # viz.text(str(pred), win='pred', opts=dict(title='Predicted'))

# test_loss/=len(test_loader.dataset)
# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))





