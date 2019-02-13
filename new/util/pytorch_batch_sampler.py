import torch 
import torch.nn as nn 
from torch.utils.data.sampler import BatchSampler as BSer
from torch.utils.data.sampler import RandomSampler as RSer
from torch.utils.data.sampler import SequentialSampler,Sampler

from torch.utils.data import Dataset,DataLoader
import numpy as np 
def main():
    test_CBatchSampler()
    
    pass
def test_CBatchSampler():
    DD = BSer(RSer(range(10)),batch_size=3,drop_last=False)
    
    # Pytorch bug
    # If you want to use the sampler
    # You need to push the data at the same time 
    # The batch sampler just grouping
    # Randomsampler shuffle the data 
    datatset = dataset_test(10) 
    dataloader = DataLoader(dataset = datatset, batch_sampler=CBatchSampler(RSer(datatset),batch_size=3,drop_last=False))
    for i in range(10):
        print(i)
        for index,enmuate in enumerate(dataloader):
            print(enmuate)
    pass
class dataset_test(Dataset):
    def __init__(self,num=10):
        self.data = np.array(range(num))
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]
class my_sampler(Sampler):
    def __init__(self,data_source):
        super(self,Sampler).__init__()
        self.data_source = data_source
        pass
    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())
        pass
    def __len__(self):
        return len(self.data_source)
        pass
class CBatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        batch_acc_record = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch_acc_record.extend(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            num_sample = self.batch_size - len(batch)
            # print("BS {}".format(self.batch_size))
            # print(len(batch))
            for index, data in enumerate (torch.randperm(len(batch_acc_record))):
                if index<num_sample:
                    batch.append(batch_acc_record[int(data)])
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

if __name__ =="__main__":
    main()