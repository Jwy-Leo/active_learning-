
import pickle
import numpy as np
import torch.utils.data as data
import torch 
import matplotlib.pyplot as plt
# DATA_ROOT = 'cifar_10/'
# Cifar  10 : wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Cifar 100 : wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
DATA_ROOT = './data/Cifar10/cifar-10-batches-py/'
def main():
    
    DL=torch.utils.data.DataLoader(ListDataloader(train = False),batch_size = 1, shuffle = True)
    for ind, (a,b) in enumerate(DL):
        import pdb;pdb.set_trace()
        print('------------------------------------------------')
        print(ind)
        print(a.shape)
        print(b)
        print('--------------------------------------------------')
class ListDataloader(data.Dataset):
    def __init__(self,train):
        self.train = train 
        if self.train == True:
            self.datalens = 50000
        else:
            self.datalens = 10000
    
    def __getitem__(self,idx):
        images, labels= GET_DATA(self.train)
        # images size 32x32x3
        
        image  = images[idx]
        label = labels[idx]
        return (image, label)
    
    def __len__(self):
        if self.train == True:
            return 50000
        else:
            return 10000
        

def GET_DATA(train):
    
    def load_file(filename):
        with open(filename, 'rb') as f:
            # Data = pickle.load(f, encoding='latin1')
            Data = pickle.load(f)
        return Data
    
    data_1 = load_file(DATA_ROOT + 'data_batch_1')
    data_2 = load_file(DATA_ROOT + 'data_batch_2')
    data_3 = load_file(DATA_ROOT + 'data_batch_3')
    data_4 = load_file(DATA_ROOT + 'data_batch_4')
    data_5 = load_file(DATA_ROOT + 'data_batch_5')
    test = load_file(DATA_ROOT + 'test_batch')

    if train == True:
        for index,(a,b) in enumerate(data_1.items()):
            # print(index,b) 
            if index == 0:
                img = b
            elif index == 1:
                label = b

        for index,(a,b) in enumerate(data_2.items()):
            if index == 0:
                img = np.vstack((img,b))
            elif index == 1:
                label = np.vstack((label,b))

        for index,(a,b) in enumerate(data_3.items()):
            if index == 0:
                img = np.vstack((img,b))
            elif index == 1:
                label = np.vstack((label,b))
    
        for index,(a,b) in enumerate(data_4.items()):
            if index == 0:
                img = np.vstack((img,b))
            elif index == 1:
                label = np.vstack((label,b))
    
        for index,(a,b) in enumerate(data_5.items()):
            if index == 0:
                img = np.vstack((img,b))
            elif index == 1:
                label = np.vstack((label,b))
    
    else:
        for index,(a,b) in enumerate(test.items()):
            if index == 0:
                img = b
            elif index == 1:
                label = b

    return (img,label)


'''
train = ListDataloader(train=False)
l = data.DataLoader(train, batch_size = 1, shuffle = True)

for ind, (a,b) in enumerate(l):
    print('------------------------------------------------')
    print(ind)
    print(a.shape)
    print(b)
    print('--------------------------------------------------')
'''
if __name__=="__main__":
    main()


