import numpy as np
import torch 
def Classification_encode_1C(img,target):
    # Database dimention N x 784 + 1 (from 28x28 img + 1x1 label) 
    # data format : |img|label|
    if type(img) is torch.Tensor:
        img = img.squeeze(dim=0).squeeze(dim=0).data.numpy()
        img_v = np.reshape(img,-1)
        img_v = img_v *255
        img_v = img_v.astype(int)
    if type(target) is torch.Tensor:
        target = target.data.numpy()
    data_vector = np.concatenate([img_v, target], axis = 0)
    return data_vector
def Classification_encode_1C_to_3C(img,target):
    # Database dimention N x 784x3 + 1 (from 28x28x3 img + 1x1 label) 
    # data format : |img|label|
    if type(img) is torch.Tensor:
        img = img.squeeze(dim=0).squeeze(dim=0).data.numpy()
        img = img[...,None]
        img = np.concatenate([img,img,img],axis = 2)
        img_v = np.reshape(img,-1)
        img_v = img_v *255
        img_v = img_v.astype(int)
    if type(target) is torch.Tensor:
        target = target.data.numpy()
    data_vector = np.concatenate([img_v, target], axis = 0)
    return data_vector
def Classification_decode(index,Database, image_size = 28):
    # Database dimention N x 784 + 1 (from 28x28 img + 1x1 label) 
    # data format : |img|label|
    import pdb;pdb.set_trace()
    imgs = Database[index,:-1]
    imgs = np.reshape(imgs, (len(index),image_size,image_size))
    labels = Database[index,-1]
    return imgs, labels

def Classification_decode_3C(index,Database,image_size = 28):
    # Database dimention N x 784x3 + 1 (from 28x28x3 img + 1x1 label) 
    # data format : |img|label|
    imgs = Database[index,:-1]
    imgs = np.reshape(imgs, (image_size,image_size,3))
    labels = Database[index,-1]
    return imgs, labels
def Detection_encode(img,target,image_size = 28):
    # Database dimention N x 784 + 1 + 4 (from 28x28 img + 1x1 label + minx,miny,maxx,maxy) 
    # data format : |img|label|bounding box ratio of images|
    if type(img) is torch.Tensor:
        img = img.squeeze(dim=0).squeeze(dim=0).data.numpy()
        [x,y] = np.where(img != 0)
        minX, minY, maxX, maxY = min(x) / float(image_size), min(y) / float(image_size), max(x) / float(image_size), max(y) / float(image_size)
        img_v = np.reshape(img,-1)
    if type(target) is torch.Tensor:
        target = target.data.numpy()
    data_vector = np.concatenate([img_v, target, [minX], [minY], [maxX],[maxY]], axis = 0)
    return data_vector
def Detection_decode(index,Database,image_size = 28):
    # Database dimention N x 784 + 1 + 4 (from 28x28 img + 1x1 label + minx,miny,maxx,maxy) 
    # data format : |img|label|bounding box ratio of images|
    imgs = Database[index,:-5]
    imgs = np.reshape(imgs, (len(index),image_size,image_size))
    bbox_information = np.zeros((1,4))
    labels = Database[index,-5]
    bbox_information = Database[index,-4:] * image_size
    return imgs, labels, bbox_information