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

train_transform=video_transforms.Compose(
    [
        video_transforms.RandomResizedCrop(160),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485,0.456,0.406),
                      (0.229,0.224,0.225))]
)

val_transform=video_transforms.Compose(
    [
        video_transforms.Resize((182,242)),
        video_transforms.CenterCrop(160),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485,0.456,0.406),
                      (0.229,0.224,0.225))]
)

train_loader=torch.utils.data.DataLoader(
    TSNDataSet("","tsntrain_01.lst",
               num_segments=2,
               new_length=16,
               modality="RGB",
               image_tmpl="frame{:06d}.jpg",
               transform=train_transform),
    batch_size=6,
    shuffle=True,
    num_workers=24,
    pin_memory=True
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
    num_workers=24,
    pin_memory=True
)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best.pth.tar')

def adjust_learning_rate(learning_rate,weight_decay,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']

def train(train_loader,net,criterion,optimizer,epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net=net.train()

    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        # inputs,labels=Variable(inputs),Variable(labels)

        outputs=net(inputs)
        loss=criterion(outputs,labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        clip_gradient = 20
        if clip_gradient is not None:
            total_norm = clip_grad_norm(net.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, clip_gradient / total_norm))
        optimizer.step()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'\
                  'lr {lr}'.format(
                   epoch, i, len(train_loader), loss=losses,
                   top1=top1, top3=top3,lr=optimizer.param_groups[0]['lr']))

    print('Finished Training')

def val(val_loader,net,criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net=net.eval()

    for i,data in enumerate(val_loader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

        outputs=net(inputs)
        loss=criterion(outputs,labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        if i % 10 == 0:
            print('Val: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(val_loader), loss=losses,
                top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return top1.avg

def main():

    base_model = P3D199(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, 101)

    num_segments = 2
    model = TSN(101,num_segments,"RGB",base_model,new_length=16)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    model = nn.DataParallel(model,device_ids=[0,1])

    policies = get_optim_policies(model)
    learning_rate = 0.001
    weight_decay = 0
    optimizer = optim.SGD(policies, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    start_epoch = 0
    epochs = 90

    best_prec1 = 0

    resume = 'checkpoint.pth.tar'
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = val(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
    main()