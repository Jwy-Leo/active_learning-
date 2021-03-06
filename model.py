import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_net(name):
    if name == 'MNIST':
        # return Net1
        # return LeNet5
        return LeNet5_dataparallel
    elif name == 'FashionMNIST':
        return Net1
    elif name == 'SVHN':
        return Net2
    elif name == 'CIFAR10':
        return Net3
class LeNet5(nn.Module):
    def __init__(self,feature_out = 84):
        super(LeNet5,self).__init__()
        self.embedding_dim = feature_out
        Module_list = []

        Module_list.append(nn.Conv2d(3,6,kernel_size = 5, stride = 1, padding = 2))
        Module_list.append(nn.BatchNorm2d(6))
        Module_list.append(nn.ReLU())
        Module_list.append(nn.MaxPool2d(2))
        Module_list.append(nn.Conv2d(6,16,kernel_size = 5, stride = 1))
        Module_list.append(nn.BatchNorm2d(16))
        Module_list.append(nn.ReLU())
        Module_list.append(nn.MaxPool2d(2))
        self.FE = nn.Sequential(*Module_list)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,self.embedding_dim)
        self.fc3 = nn.Linear(self.embedding_dim,10)
    def forward(self,x):
        x = self.FE(x)
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        feature = torch.relu(self.fc2(x))
        x = self.fc3(feature)
        return x, feature
    def get_embedding_dim(self):
        return self.embedding_dim
class LeNet5_dataparallel(nn.DataParallel):
    def __init__(self,module=LeNet5(),device_ids=None,output_device=None,dim=0):
        super(LeNet5_dataparallel,self).__init__(module,device_ids,output_device,dim)
    def get_embedding_dim(self):
        return self.module.get_embedding_dim()
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
