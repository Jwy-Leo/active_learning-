import sys
import os
sys.path.append(os.getcwd()+"/src")
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import torch 
import sys
import random
# sys.path.append("../util/")
# from util_Mnist_train_to_npz import Classification_decode_3C as DATA_decode
# from util_Mnist_train_to_npz import Classification_decode_3C_M as DATA_decode_M
from util.util_Mnist_train_to_npz import Classification_decode_3C as DATA_decode
# from util.util_Mnist_train_to_npz import Classification_decode_3C_M as DATA_decode_M
import numpy as np
def main():
    
    # dataloader = DataLoader(dataset = mnist_classification_three_channel(
    #    np.load('./data/mnist-m/mnist-m_test.npy')),batch_size = 2,shuffle = False)
    # dataloader = DataLoader(dataset = mnist_classification_three_channel(
    #      np.load('./data/mnist/Mnist_Database_classification_eval_3C.npy')),batch_size = 200,shuffle = False)
    # for index,(imgs,label) in enumerate(dataloader):
    #     pass
    
    dataloader = DataLoader(dataset = Cifar10_classification_three_channel(
         np.load('./data/Cifar10/cifar-10-batches-py/Cifar10_train_50000_3C.npy').item()['data']),batch_size = 200,shuffle = False)
    for index,(imgs,label) in enumerate(dataloader):
        import pdb;pdb.set_trace()
        pass
# class mnist_classification_three_channel(Dataset):
#     def __init__(self,npy_database,transform=None):
#         self.npy_database = npy_database    # BN x 28*28*3+1
#         if transform != None:
#             self.transform = transform
#         else:
#             self.transform = transforms.ToTensor()
#     def __getitem__(self,index):
#         img, label = DATA_decode(index,self.npy_database)
#         import pdb
#         pdb.set_trace()
#         img = self.transform(img)
        
#         label = torch.LongTensor(np.array([label]))
#         # torch.one_hot
#         # label = torch.zeros((1,10)).type(torch.LongTensor)
#         # label[0,label]=1
#         return img, label
#     def __len__(self):
#         return self.npy_database.shape[0]
class mnist_classification_three_channel(Dataset):
    def __init__(self,npy_database,transform=None):
        self.npy_database = npy_database    # BN x 28*28*3+1
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
    def __getitem__(self,index):
        # img, label = DATA_decode_M(index,self.npy_database)
        img, label = DATA_decode(index,self.npy_database)
        img = img.astype(np.uint8)
        # img = img.transpose((2,0,1))
        # img = torch.FloatTensor(img)
        img = self.transform(img)
        label = torch.LongTensor(np.array([label]))
        # torch.one_hot
        # label = torch.zeros((1,10)).type(torch.LongTensor)
        # label[0,label]=1
        return img, label
    def __len__(self):
        return self.npy_database.shape[0]
class Random_choice_with_Random_compose(object):
    def __init__(self,transforms,p):
        self.transforms = transforms
        self.p = p
    def __call__(self,img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            if self.p >= random.random():
                img = self.transforms[i](img)
        return img
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class mnist_classification_with_augmentation(mnist_classification_three_channel):
    def __init__(self,npy_database,transform=None):
        arguments = (npy_database,transform)
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
            self.transform_after = Random_choice_with_Random_compose(
                [transforms.ColorJitter(brightness = 0.05,contrast = 0.05,saturation =0.05,hue=0.05),               # HSV and contrast
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAffine(degrees=(-10,10),translate=(0.2,0.2),scale=(0.7,1.3),shear=(-10,10))],   # -10~10 degree
                p=0.5
            )
        super(mnist_classification_three_channel,self).__init__(*arguments)
        
    def __getitem__(self,index):
        
        img, label = DATA_decode(index,self.npy_database)
        img = img.astype(np.uint8)
        # "ColorJitter", "RandomRotation","RandomGrayscale","LinearTransformation"
        # Imaging Color chang, Rotation, Grayscale

        img = self.transform(img)
        img = self.transform_after(img)
        label = torch.LongTensor(np.array([label]))
        
        return img, label


class Cifar10_classification_three_channel(Dataset):
    def __init__(self,npy_database,transform=None):
        self.npy_database = npy_database    # BN x 32*32*3+1
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
    def __getitem__(self,index):
        # img, label = DATA_decode_M(index,self.npy_database)
        
        img, label = DATA_decode(index,self.npy_database,image_size=32)
        
        img = img.astype(np.uint8)
        
        # img = img.transpose((2,0,1))
        # img = torch.FloatTensor(img)
        img = self.transform(img)
        label = torch.LongTensor(np.array([label]))
        # torch.one_hot
        # label = torch.zeros((1,10)).type(torch.LongTensor)
        # label[0,label]=1
        return img, label
    def __len__(self):
        return self.npy_database.shape[0]
if __name__=="__main__":
    main()
