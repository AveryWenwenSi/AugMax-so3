'''
Training with AugMax data augmentation 
'''
import os, sys, argparse, time, random
sys.path.append('./')
import numpy as np 

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import numpy as np 
from PIL import Image


from augmax_modules import augmentations
from augmax_modules.augmax import AugMaxDataset, AugMaxModule, AugMixModule

from models.cifar10.resnet_DuBIN import ResNet18_DuBIN
from models.cifar10.wideresnet_DuBIN import WRN40_DuBIN
from models.cifar10.resnext_DuBIN import ResNeXt29_DuBIN

from models.imagenet.resnet_DuBIN import ResNet18_DuBIN as INResNet18_DuBIN
from models.imagenet.resnet_DuBIN import ResNet50_DuBIN as INResNet50_DuBIN

from dataloaders.cifar10 import cifar_dataloaders

from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders, tiny_imagenet_deepaug_dataloaders
from dataloaders.imagenet import imagenet_dataloaders, imagenet_deepaug_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.attacks import AugMaxAttack, FriendlyAugMaxAttack

import os
import numpy as np
import pdb
from liegroups.torch import SE2, SO2, utils, SO3, SE3

add_noise = True
use_so3 = False
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier')
parser.add_argument('--gpu', default='2,3')
parser.add_argument('--num_workers', '--cpus', default=16, type=int)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'tin', 'IN'], help='which dataset to use')
parser.add_argument('--data_root_path', '--drp', help='Where you save all your datasets.')
parser.add_argument('--model', '--md', default='WRN40', choices=['ResNet18', 'ResNet50', 'WRN40', 'ResNeXt29'], help='which model to use')
parser.add_argument('--widen_factor', '--widen', default=2, type=int, help='widen factor for WRN')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--decay_epochs', '--de', default=[100,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training.')
parser.add_argument('--test_batch_size', '--tb', type=int, default=512, help='Batch size for validation.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# AugMix options
parser.add_argument('--mixture_width', default=3, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture_depth', default=-1, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug_severity', default=3, help='Severity of base augmentation operators')
# augmax parameters:
parser.add_argument('--attacker', default='fat', choices=['pgd', 'fat'], help='How to solve the inner maximization problem.')
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--alpha', type=float, default=0.1, help='attack step size')
parser.add_argument('--tau', type=int, default=1, help='Early stop iteration for FAT.')
parser.add_argument('--steps', type=int, default=5, help='The maximum iteration for the attack (FAT/PGD).')
parser.add_argument('--Lambda', type=float, default=10, help='Trade-off hyper-parameter in loss function.')
# others:
parser.add_argument('--deepaug', action='store_true', help='If true, use deep augmented training set. (Only works for ImageNet.)')
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--save_root_path', '--srp', default='ssd1/', help='where you save the outputs')
# DDP settings:
parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
parser.add_argument('--node_id', default=0, type=int, help='Node ID')
parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
args = parser.parse_args()
# adjust learning rate:
if args.dataset == 'tin':
    args.lr *= args.batch_size / 256. # linearly scaled to batch size
    augmentations.IMAGE_SIZE = 64 # change imange size
elif args.dataset == 'IN':
    # args.cpus = 12
    args.lr *= args.batch_size / 256. * 3 # linearly scaled to batch size
    augmentations.IMAGE_SIZE = 224 # change imange size

# set CUDA:
# if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'

# select model_fn:
if args.dataset == 'IN':
    if args.model == 'ResNet18':
        model_fn = INResNet18_DuBIN
    elif args.model == 'ResNet50':
        model_fn = INResNet50_DuBIN
else:
    if args.model == 'ResNet18':
        model_fn = ResNet18_DuBIN
    elif args.model == 'WRN40':
        model_fn = WRN40_DuBIN
    elif args.model == 'ResNeXt29':
        model_fn = ResNeXt29_DuBIN

# mkdirs:
model_str = model_fn.__name__
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-' + '-'.join(map(str, args.decay_epochs)) 
loss_str = 'Lambda%s' % args.Lambda
attack_str = ('%s-%s' % (args.attacker, args.tau) if args.attacker == 'fat' else args.attacker) + '-' + ('targeted' if args.targeted else 'untargeted') + '-%d-%s' % (args.steps, args.alpha)
if args.deepaug:
    dataset_str = '%s_deepaug' % args.dataset
    assert args.dataset in ['tin', 'IN']
else:
    dataset_str = args.dataset
if add_noise:
    if use_so3:
        save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, 'so3_noise0.5')
    else:
        # save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, 'noise0.5')
        save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, 'se3')
else:
    if use_so3:
        save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, 'so3')
    else:
        save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, 'rotation_crop')
        # save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, '%s_%s_%s_%s' % (attack_str, loss_str, opt_str, decay_str))
create_dir(save_folder)
print('saving to %s' % save_folder)

def cifar_random_affine_test_set(data_dir, num_classes=10):
    assert num_classes in [10, 100]
    CIFAR = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    test_transform = transforms.Compose(
        # [transforms.RandomAffine(degrees=30), transforms.ToTensor()])
        [transforms.RandomAffine(degrees=30, translate=(3/32,3/32)), transforms.ToTensor()])

    test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
    
    return test_data

def val_cifar_worst_of_k_affine(K, model):
    '''
    Test model robustness against spatial transform attacks using worst-of-k method on CIFAR10/100.
    '''
    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():        
        K_loss = torch.zeros((K, args.test_batch_size)).cuda()
        K_logits = torch.zeros((K, args.test_batch_size, 10)).cuda()
        for k in range(K):
            random.seed(k+1)
            val_data = cifar_random_affine_test_set(data_dir=args.data_root_path, num_classes=10)
            test_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            images, targets = next(iter(test_loader))
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction='none')
            # stack all losses:
            K_loss[k,:] = loss # shape=(K,N)
            K_logits[k,...] = logits
        # print('K_loss:', K_loss[:,0:3], K_loss.shape)
        adv_idx = torch.max(K_loss, dim=0).indices
        logits_adv = torch.zeros_like(logits).to(logits.device)
        for n in range(images.shape[0]):
            logits_adv[n] = K_logits[adv_idx[n],n,:]
        print('logits_adv:', logits_adv.shape)
        pred = logits_adv.data.max(1)[1]
        print('pred:', pred.shape)
        acc = pred.eq(targets.data).float().mean()
        # append loss:
        test_acc_meter.append(acc.item())
    print('worst of %d test time: %.2fs' % (K, time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'worst of %d: %.4f' % (K, test_acc)
    print(clean_str)

    return test_acc

def setup(rank, ngpus_per_node):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def spatial_sampler(x, theta):
    grid = nn.functional.affine_grid(theta, size=x.size())
    sampled_data = nn.functional.grid_sample(x, grid)

    return sampled_data

def train(gpu_id, ngpus_per_node):
    
    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    if gpu_id == 0:
        print(args)

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True # set cudnn.benchmark in each worker, as done in https://github.com/pytorch/examples/blob/b0649dcd638eb553238cdd994127fd40c8d9a93a/imagenet/main.py#L199

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data loader:
    if args.dataset in ['cifar10', 'cifar100']:
        num_classes=10 if args.dataset == 'cifar10' else 100
        init_stride = 1
        train_data, val_data = cifar_dataloaders(data_dir=args.data_root_path, num_classes=num_classes,
            AugMax=None, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
    elif args.dataset == 'tin':
        num_classes, init_stride = 200, 2
        train_data, val_data = tiny_imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200'),
            AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
        if args.deepaug:
            edsr_data = tiny_imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200-DeepAug-EDSR'),
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            cae_data = tiny_imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200-DeepAug-CAE'),
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
    elif args.dataset == 'IN':
        num_classes, init_stride = 1000, None
        train_data, val_data = imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet'), 
            AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
        if args.deepaug:
            edsr_data = imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet-DeepAug-EDSR'), 
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            cae_data = imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet-DeepAug-CAE'), 
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model:
    if args.dataset == 'IN':
        if args.model == 'WRN40':
            model = model_fn(widen_factor=args.widen_factor).to(device)
        else:
            model = model_fn().to(device)
    else:
        if args.model == 'WRN40':
            model = model_fn(num_classes=num_classes, init_stride=init_stride, widen_factor=args.widen_factor).to(device)
        else:
            model = model_fn(num_classes=num_classes, init_stride=init_stride).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False, find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model)

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # load ckpt:
    if args.resume:
        last_epoch, best_SA, training_loss, val_SA \
            = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_SA = 0
        # training curve lists:
        training_loss, val_SA = [], []
        
    theta_list = []
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
        start_time = time.time()

        ## training:
        model.train()
        requires_grad_(model, True)
        accs, accs_augmax, losses = AverageMeter(), AverageMeter(), AverageMeter()
        for i, (images_tuples, labels) in enumerate(train_loader):
            if (images_tuples.shape[0] != args.batch_size):
                continue
            images_tuples = images_tuples.to(device)
            labels = labels.to(device)
            # get batch:

            if i % 1 == 0:
                w = Variable(torch.zeros([1, 6]), requires_grad=True)
                
            w.retain_grad()
            if use_so3:
                theta = SO3.exp(w).mat[:2].to(device)
            else:
                # rot, trans = SE2.exp(w)[0], torch.clamp(SE2.exp(w)[1], -3, 3.)
                # theta = torch.cat([rot, trans], 1).to(device)
                rot, trans = SE3.exp(w)[0], torch.clamp(SE3.exp(w)[1], -3, 3.)
                theta = torch.cat([rot, trans], 1).to(device)
            
            theta.retain_grad()

            trans_data = spatial_sampler(images_tuples, theta.repeat(images_tuples.shape[0],1,1))
            logits = model(trans_data)

            # loss:
            loss1 = F.cross_entropy(logits, labels)
            loss1.backward()
            grad = w.grad
            # apply FGSM on affine matrix
            data_combined, target_combined = images_tuples, labels
            for inter in range(5):
                if add_noise:
                    w = w + 0.5 * (grad + torch.rand(grad.shape))
                else:
                    w = w + 0.5 * grad
                # clamp rotation here
                ww = w.clone()
                if use_so3:
                    # ww = w % 3.14
                    # ww = torch.clamp(w, -0.53, 0.53)
                    theta = SO3.exp(ww).mat[:2].to(device)
                else:
                    ww[0] = w[0] % 3.14
                    ww[0] = torch.clamp(w[0], -0.53, 0.53)
                    rot, trans = SE3.exp(ww)[0], torch.clamp(SE3.exp(ww)[1], -3., 3.)
                    theta = torch.cat([rot, trans], 1).to(device)
                    # rot, trans = SE2.exp(ww)[0], torch.clamp(SE2.exp(ww)[1], -3., 3.)
                    # theta = torch.cat([rot, trans], 1).to(device)
                theta_list.append(ww.detach().cpu().numpy())
                
                trans_data = spatial_sampler(images_tuples, theta.repeat(images_tuples.shape[0],1,1))
                data_combined = torch.cat([data_combined, trans_data], 0)
                target_combined = torch.cat([target_combined, labels], 0)
            
            # # metrics:
            optimizer.zero_grad()
            logits = model(data_combined)
            loss = F.cross_entropy(logits, target_combined)
            loss.backward()
            optimizer.step()

            l = logits.shape[0]
            accs.append((logits[:l//6].argmax(1) == labels).float().mean().item())
            accs_augmax.append((logits[l//6:].argmax(1) == target_combined[l//6:]).float().mean().item())
            losses.append(loss.item())

            if i % 50 == 0:
                train_str = 'Epoch %d-%d | Train | Loss: %.4f (%.4f, %.4f), SA: %.4f, RA: %.4f' % (epoch, i, losses.avg, loss, loss, accs.avg, accs_augmax.avg)
                if gpu_id == 0:
                    print(train_str)
                # print(theta[:,:,2].max(0))
        # lr schedualr update at the end of each epoch:
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if rank == 0:
            model.eval()
            requires_grad_(model, False)
            print(model.training)

            # eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*args.epochs)) # boolean
            eval_this_epoch = True
            
            if eval_this_epoch:
                val_SAs = AverageMeter()
                if 'DuBN' in model_fn.__name__ or  'DuBIN' in model_fn.__name__: 
                    model.apply(lambda m: setattr(m, 'route', 'M')) # use main BN
                test_acc = val_cifar_worst_of_k_affine(10, model)
                for i, (imgs, labels) in enumerate(val_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    # logits for clean imgs:
                    logits = model(imgs)
                    val_SAs.append((logits.argmax(1) == labels).float().mean().item())

                val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f' % (
                    epoch, (time.time()-start_time), current_lr, val_SAs.avg)
                print(val_str)
                fp.write(val_str + '\n')

            # save loss curve:
            training_loss.append(losses.avg)
            plt.plot(training_loss)
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'training_loss.png'))
            plt.close()

            val_SA.append(test_acc) 
            plt.plot(val_SA, 'r')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_SA.png'))
            plt.close()

            # save pth:
            if eval_this_epoch:
                # if val_SAs.avg >= best_SA:
                if test_acc >= best_SA: 
                    best_SA = test_acc
                    torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
            save_ckpt(epoch, model, optimizer, scheduler, best_SA, training_loss, val_SA, 
                os.path.join(save_folder, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

if __name__ == '__main__':
    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0)