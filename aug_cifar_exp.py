import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from my_cifar10 import CIFAR10
from torch.utils.data import random_split
from models.cifar10.resnext_DuBIN import ResNeXt29_DuBIN

import os
import numpy as np
import pdb
from liegroups.torch import SE2, SO2, utils


os.environ["TORCH_HOME"] = "./data"

device_no = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
device = torch.device(device_no if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
batch_size = 128
src_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
src_set_len = len(src_set)
src_train, src_val = random_split(src_set, [int(src_set_len*0.8), src_set_len - int(src_set_len*0.8)])
src_trainloader = torch.utils.data.DataLoader(src_train, batch_size=batch_size, shuffle=True, num_workers=2)
src_valloader = torch.utils.data.DataLoader(src_val, batch_size=batch_size, shuffle=True, num_workers=2)

tar_set_clean = CIFAR10(root='./data', train=False, download=True, transform=transform)
print("target domain size: ", len(tar_set_clean))
# debug with same val and test
tar_testloader_clean = torch.utils.data.DataLoader(tar_set_clean, batch_size=batch_size, shuffle=False, num_workers=2)
tar_set = CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(3/32,3/32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]))
tar_testloader = torch.utils.data.DataLoader(tar_set, batch_size=batch_size, shuffle=False, num_workers=2)
eps = 0.02  # theta step size

def spatial_sampler(x, theta):
    grid = nn.functional.affine_grid(theta, size=x.size())
    sampled_data = nn.functional.grid_sample(x, grid)

    return sampled_data

net = nn.DataParallel(ResNeXt29_DuBIN()).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)

net.train()

def train(epoch):
    trans = torch.zeros([2, 1])
    for batch_idx, (data, target) in enumerate(src_trainloader):
        if batch_idx % 1 == 0:
            w = Variable(torch.Tensor([0.08]), requires_grad=True)
            # w = Variable(torch.Tensor([0.,0.,0.]), requires_grad=True)
        w.retain_grad()
        # rot = SO2.exp(w).mat
        # theta = torch.cat([rot, trans], 1).to(device)
        # theta.retain_grad()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # trans_data = spatial_sampler(data, theta.repeat(data.shape[0],1,1))
        # trans_data1 = trans_data.detach().cpu().cuda()
        # target_combined = torch.cat([target, target,target], 0)

        # output = net(trans_data)
        output = net(data)

        loss = nn.functional.cross_entropy(output, target)
        # loss.backward()
        # grad = w.grad
        # apply FGSM on affine matrix
        # w = w + eps * grad
        
        # this time update net with the new transferred images
        # optimizer.zero_grad()
        # rot = SO2.exp(w).mat
        # theta = torch.cat([rot, trans], 1).to(device)
        # trans_data2 = spatial_sampler(data, theta.repeat(data.shape[0],1,1))
        # data_combined = torch.cat([data, trans_data1, trans_data2], 0)
        # output = net(data_combined)
        # loss = nn.functional.cross_entropy(output, target_combined)
        loss.backward()
        optimizer.step()

    # print(eps * grad)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(src_trainloader.dataset),
                       100. * batch_idx / len(src_trainloader), loss.item()))

    net.eval()
    t_test_loss = 0
    n_correct = 0

    # for data, target in tar_testloader:
    #     data, target = data.to(device), target.to(device)
    #     output = net(data).detach().cpu()
    #     pred = output.max(1, keepdim=True)[1]
    #     n_correct += (pred.flatten() == target.detach().cpu()).sum()
    # avg_loss = n_correct / len(tar_testloader) / batch_size
    # print("Aug pert test_acc : ", avg_loss.item())

    t_test_loss = 0
    n_correct = 0
    for data, target in tar_testloader_clean:
        data, target = data.to(device), target.to(device)
        output = net(data).detach().cpu()
        pred = output.max(1, keepdim=True)[1]
        n_correct += (pred.flatten() == target.detach().cpu()).sum()

    avg_loss = n_correct / len(tar_testloader) / batch_size
    print("Aug clean test_acc : ", avg_loss.item())
    return None


def test(loader):
    net = nn.DataParallel(ResNeXt29_DuBIN()).to(device)
    net.load_state_dict(torch.load("temp_model"))
    net.eval()
    t_test_loss = 0
    n_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        test_loss = nn.functional.cross_entropy(output, target)
        t_test_loss += test_loss
        # print("test_loss : ", test_loss.item())
        pred = output.max(1, keepdim=True)[1]
        n_correct += (pred.flatten() == target).sum()

    avg_loss = n_correct / len(loader) / batch_size
    print("Aug 5p test_acc : ", avg_loss.item())


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(theta, epoch, net):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(tar_testloader_clean))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = spatial_sampler(data, theta.repeat(data.shape[0],1,1)).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
        # plt.show()
        plt.savefig('vis_{}.png'.format(epoch))


for epoch in range(1, 100 + 1):
    theta = train(epoch)
    plt.close('all')
    # test(tar_testloader)
    # test(tar_testloader_clean)
    # Visualize the STN transformation on some input batch
    # visualize_stn(theta, epoch)

# plt.show()
