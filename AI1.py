from PIL import Image
from torchvision import models
from torchvision import transforms
import torch
import numpy as np
import cv2

def Erosion(dress):
    src = cv2.imread(dress,cv2.IMREAD_UNCHANGED)

    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.dilate(src,kernel,2)

    img = Image.fromarray(erosion)
    return img

resnet = models.resnet101(pretrained=True)
preprocess = transforms.Compose([           
    #transforms.Resize(256),                     #更改图像大小
    #transforms.CenterCrop(244),                 #更改裁剪的范围，通过改变这两项，可以使模型的输出错误
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229,0.224,0.225])])

#image = Image.open("D:/torch/bobby.jpg")

image = Erosion("D:/torch/bobby.jpg")

img_t = preprocess(image)

batch_t = torch.unsqueeze(img_t,0)

resnet.eval()

out = resnet(batch_t)

with open('D:/torch/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_,index = torch.max(out,1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])