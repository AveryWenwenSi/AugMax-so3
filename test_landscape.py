import os, sys, argparse, time, random, dill
sys.path.append('./')
import numpy as np 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.imagenet import imagenet_dataloaders

from utils.utils import *
import imagenet_models as models


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier')
parser.add_argument('--gpu', default='1')
parser.add_argument('--cpus', type=int, default=4)
# dataset:
parser.add_argument('--data_root_path', '--drp', default='/data5/wenwens/', help='Where you save all your datasets.')
parser.add_argument('--test_batch_size', '--tb', type=int, default=256, help='Batch size for validation.')
parser.add_argument('--mixture_width', default=3, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture_depth', default=-1, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug_severity', default=3, help='Severity of base augmentation operators')
parser.add_argument('--k', default=10, type=int, help='hyperparameter k in worst-of-k spatial attack')
parser.add_argument('--save_root_path', '--srp', default='ssd1/', help='where you save the outputs')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

num_classes, init_stride = 1000, None

model = models.resnet18(num_classes=num_classes, pretrained=False)

resume_path = 'out/random_30/checkpoint.pt.best'
checkpoint = torch.load(resume_path, pickle_module=dill)
state_dict_path = 'model'
if not ('model' in checkpoint):
    state_dict_path = 'state_dict'

sd = checkpoint[state_dict_path]
sdd = {}
for k,v in sd.items():
    if k.startswith("module.model."):
        sdd[k[len('module.model.'):]]=v 
model.load_state_dict(sdd)
model = torch.nn.DataParallel(model)
model = model.cuda()

def imagenet_random_affine_test_set(data_dir, num_classes=1000):
    val_data = imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet'), 
        AugMax=None, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity,
        worst_of_k=True
    )
    return val_data

def val_imagenet_worst_of_k_affine(K):
    '''
    Test model robustness against spatial transform attacks using worst-of-k method on CIFAR10/100.
    '''
    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():        
        K_loss = torch.zeros((K, args.test_batch_size)).cuda()
        K_logits = torch.zeros((K, args.test_batch_size, num_classes)).cuda()
        for k in range(K):
            random.seed(k+1)
            val_data = imagenet_random_affine_test_set(data_dir=args.data_root_path, num_classes=num_classes)
            test_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.cpus, pin_memory=True)
            images, targets = next(iter(test_loader))
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction='none')
            # stack all losses:
            K_loss[k,:] = loss # shape=(K,N)
            K_logits[k,...] = logits
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

val_imagenet_worst_of_k_affine(args.k)