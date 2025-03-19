import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
from functools import partial
import json

import torch.utils.data as data
from torchvision import transforms

from datasets.cifar10 import CIFAR10_truncated
from datasets.cifar100 import CIFAR100_truncated
from datasets.fmnist import FashionMNIST_truncated
from datasets.folder import ImageFolder_custom
from datasets.wrapper import AugmentedDatasetWrapper

from models import resnet_cifar_fa

def init_logger(args):
    os.makedirs(args.logdir, exist_ok=True)
    log_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    if args.log_file_name is None:
        log_file_name = f'experiment-arguments_{log_file_name}'
    else:
        log_file_name = args.log_file_name
    with open(os.path.join(args.logdir, log_file_name + '.json'), 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_file_name + '.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()            
    console_handler.setLevel(logging.INFO)             
    console_formatter = logging.Formatter(
        '%(asctime)s %(message)s',
        datefmt='%m-%d %H:%M'
    )                                                   
    console_handler.setFormatter(console_formatter)       
    logger.addHandler(console_handler)    

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return logger, log_file_name

def get_global_dataset(args):
    if args.dataset == 'fmnist':
        normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])
        
        transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        # test set data prep
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
        train_ds = FashionMNIST_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = FashionMNIST_truncated(args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        train_ds = CIFAR10_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR10_truncated(args.datadir, train=False, transform=transform_test, download=True)
    
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        train_ds = CIFAR100_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR100_truncated(args.datadir, train=False, transform=transform_test, download=True)
    
    return train_ds, val_ds, test_ds

def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.debug('Data statistics: %s' % str(net_cls_counts))
    return

def partition_data(global_train_dataset, args, logger):
    X_train, y_train = global_train_dataset.data, global_train_dataset.target

    n_train = y_train.shape[0]

    if args.partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_clients)}

    elif args.partition == "noniid":
        min_size = 0

        K = global_train_dataset.num_classes
        N = len(global_train_dataset)

        net_dataidx_map = {}

        while min_size < args.min_require_size:
            idx_batch = [[] for _ in range(args.n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.beta, args.n_clients))
                proportions = np.array([p * (len(idx_j) < N / args.n_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # non-iid balanced
    elif args.partition == "noniid_balanced":
        K = global_train_dataset.num_classes
        N = len(global_train_dataset)

        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(args.n_clients)}
        assigned_ids = []
        idx_batch = [[] for _ in range(args.n_clients)]
        num_data_per_client=int(N/args.n_clients)
        for i in range(args.n_clients):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(args.beta, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k]=proportions[k]
            weights[assigned_ids] = 0.0
            idx_batch[i] = (torch.multinomial(weights, num_data_per_client, replacement=False)).tolist()
            assigned_ids+=idx_batch[i]

        for j in range(args.n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    record_net_data_stats(y_train, net_dataidx_map, logger)
    return net_dataidx_map

def shuffle_clients(args):
    n_party_per_round = int(args.n_clients * args.sample_fraction)
    party_list = [i for i in range(args.n_clients)]
    party_list_rounds = []
    if n_party_per_round != args.n_clients:
        for i in range(args.round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.round):
            party_list_rounds.append(party_list)
    return party_list_rounds

def get_client_datasets(global_train_dataset, client_data_map, args):
    client_datasets = {}
    for i in range(args.n_clients):    
        client_datasets[i] = (data.Subset(global_train_dataset, client_data_map[i]))

    return client_datasets

def get_client_meta_datasets(client_datasets, args):
    client_meta_datasets = {}
    transform = []
    for i in range(args.n_clients):
        client_meta_datasets[i] = (AugmentedDatasetWrapper(client_datasets[i], transform=transform))

    return client_meta_datasets


def get_global_dataloader(global_train_dataset, global_val_dataset, global_test_dataset, args):
    global_train_dataloader = data.DataLoader(dataset=global_train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,num_workers=args.num_workers)
    global_val_dataloader = None
    if global_val_dataset is not None:
        global_val_dataloader = data.DataLoader(dataset=global_val_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    global_test_dataloader = data.DataLoader(dataset=global_test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return global_train_dataloader, global_val_dataloader, global_test_dataloader

def get_client_dataloaders(client_datasets, args):
    dataloaders = {}
    for i in range(args.n_clients):
        client_train_dataloader = data.DataLoader(dataset=client_datasets[i], batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,num_workers=args.num_workers)
        dataloaders[i] = client_train_dataloader

    return dataloaders

def get_client_meta_dataloaders(client_datasets, args):
    dataloaders = {}
    # worker_init = partial(worker_init_fn, seed=args.seed)
    # for i in range(args.n_clients):
    #     # TODO: sampler? batch sampler? collate?
    #     client_train_dataloader = data.DataLoader(dataset=client_datasets[i], batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,num_workers=args.num_workers, worker_init_fn=worker_init, sampler=, batch_sampler=,collate_fn=)
    #     dataloaders[i] = client_train_dataloader

    return dataloaders

def init_nets(dataset, num_nets, args, device='cpu', base=False):
    nets = {}
    num_classes = dataset.num_classes
    norm_layer = None
    if args.group_norm:
        norm_layer = lambda num_channels: nn.GroupNorm(num_groups=args.num_groups, num_channels=num_channels)

    for net_i in range(num_nets):
        if args.model == 'resnet50':
            if base:
                net = resnet_cifar_fa.ResNet50_cifar10_fa(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, pre_fa=False, post_fa=False)
            else:    
                net = resnet_cifar_fa.ResNet50_cifar10_fa(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, pre_fa=args.pre_fa, post_fa=args.post_fa)
        else:
            raise ValueError('wrong model config')
        net.to(device)
        nets[net_i] = net

    return nets

def compute_accuracy(model, dataloader, device):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #print("x:",x)
            x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            _,out = model(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()

    return correct / float(total)

def avg_last_n(accuracy_list, n):
    if not accuracy_list:
        return None
    recent = accuracy_list[-n:] if len(accuracy_list) >= n else accuracy_list
    return sum(recent) / len(recent)