import torch
import torchvision.datasets as dest
import torchvision.transforms as trans
from torch.autograd import Variable
import time

# Load Data set
mnistTrain = dest.MNIST(root="~/TestSet/MNIST",
                        train=True,
                        transform=trans.transforms.ToTensor(), download=True)

mnistTest = dest.MNIST(root="~/TestSet/MNIST",
                       train=False,
                       transform=trans.transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(mnistTrain,batch_size=32)
# test_loader = torch.utils.data.DataLoader(mnistTest,batch_size=32)

n_input = 28*28
n_hidden = 28*28*5
n_output = 10

# Define Net


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 16 * 14 * 14
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.hidden = torch.nn.Linear(7*7*64,1024)

        self.out = torch.nn.Linear(1024,10)

    def forward(self, in_data):
        in_data, = self.conv1(in_data),
        in_data, = self.conv2(in_data),
        # in_data = in_data[0]
        in_data = in_data.view(in_data.size(0),-1)
        out = self.hidden(in_data)
        out = self.out(out)
        return out


net = Net().cuda()
print (net)

optimizer = torch.optim.SGD(net.parameters(), 0.001)
loss_function = torch.nn.CrossEntropyLoss()

# Train
num_epoches = 10000

for epoch in range(num_epoches):

    running_Loss = 0.0
    running_acc = 0
    time_s = time.time()

    for i, (img, label) in enumerate(train_loader, 1):

        # Forward
        label = Variable(label).cuda()
        img = Variable(img).cuda()

        out = net(img)
        loss = loss_function(out, label)

        running_Loss += loss
        _, pred = torch.max(out,1)

        running_acc += (pred == label).cpu().numpy().sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_e = time.time()
    torch.save(net,'1.pkl')
    print('[{0}/{1}, Loss: {2:<5}\t Acc: {3:<5}\t Cost: {4:<5}s ]'.format(epoch+1,num_epoches,running_Loss,running_acc/train_loader.__len__(), time_e - time_s))

