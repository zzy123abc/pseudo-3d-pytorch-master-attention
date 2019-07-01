import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import os
import shutil
import math
import numpy as np
from PIL import Image

from tsn_dataset import TSNDataSet
from p3d_model import P3D199,get_optim_policies
import video_transforms

from tsn_models import TSN
from torch.nn.utils import clip_grad_norm

val_transform=video_transforms.Compose(
    [
        video_transforms.Resize((182,242)),
        video_transforms.CenterCrop(160),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485,0.456,0.406),
                      (0.229,0.224,0.225))]
)

val_loader=torch.utils.data.DataLoader(
    TSNDataSet("","tsntest_01.lst",
               num_segments=2,
               new_length=16,
               modality="RGB",
               image_tmpl="frame{:06d}.jpg",
               transform=val_transform,
               random_shift=False),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=False
)

if __name__ == '__main__':

    base_model = P3D199(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, 101)

    model = TSN(101,2,"RGB",base_model,new_length=16)

    model = nn.DataParallel(model,device_ids=[0,1])

    resume = 'best.pth.tar'
    if os.path.isfile(resume):
        checkpoint = torch.load(resume,map_location={'cuda:0':'cpu'})
        model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()
    res = []

    for i,data in enumerate(val_loader,0):
        inputs,labels=data
        inputs, labels = Variable(inputs), Variable(labels)

        out=model(inputs)
        m=nn.Softmax()
        out=m(out)
        max,maxindex=torch.max(out,1)
        print(maxindex.data.numpy()[0])
        res.append(maxindex.data.numpy()[0])

    np.save('res', np.array(res))
