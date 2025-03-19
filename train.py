import torch
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np

from algs.decorr import FedDecorrLoss
from fa.fa_conv import FeedbackConvLayer
from fa.fa_linear import FeedbackLinearLayer

def fedavg(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()

    total_loss = 0.0
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            loss = criterion(out, target)
            total_loss += loss.item()
            
            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()
            # if args.pre_fa or args.post_fa and (args.scale_conv or args.scale_linear):
            if not args.no_scale:
                if args.pre_fa or args.post_fa:
                    with torch.no_grad():
                        for m in net.modules():
                            if isinstance(m, FeedbackConvLayer) or isinstance(m, FeedbackLinearLayer):
                                m.scale_B()

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs

def fedprox(net, global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    global_weight_collector = list(global_model.parameters())

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            loss = criterion(out, target)

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if args.pre_fa or args.post_fa:
                with torch.no_grad():
                    for m in net.modules():
                        if isinstance(m, FeedbackConvLayer) or isinstance(m, FeedbackLinearLayer):
                            m.scale_B()
    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs

def moon(net, global_model, previous_net, train_dataloader, optimizer, device, args):
    total_loss = 0.
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    cos=torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features1, out = net(x)
            features2, _ = global_model(x)
            
            posi = cos(features1, features2)
            logits = posi.reshape(-1,1)

            features3, _ = previous_net(x)
            nega = cos(features1, features3)
            logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= args.temperature
            labels = torch.zeros(x.size(0)).long().to(device)
            loss2 = args.mu * criterion(logits, labels)
            loss1 = criterion(out, target)
            loss = loss1 + loss2
            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features1)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()

            if args.pre_fa or args.post_fa:
                with torch.no_grad():
                    for m in net.modules():
                        if isinstance(m, FeedbackConvLayer) or isinstance(m, FeedbackLinearLayer):
                            m.scale_B()
    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs

def adjust_lr(round, current_lr, args):
    if args.scheduler == 'linear':
        new_lr = args.eta_min + (args.lr - args.eta_min) * (1 - round / args.round)
    elif args.scheduler == 'cosine':
        new_lr = args.eta_min + (args.lr - args.eta_min) * 0.5 * (1 + math.cos(math.pi * round / args.round))
    elif args.scheduler == 'step':
        if (round + 1) in args.schedule_round:
            new_lr = current_lr * args.lr_gamma
        else:
            new_lr = current_lr
    else:
        new_lr = current_lr
    return new_lr

def train_local_net(dataloaders, nets, global_model, prev_nets, device, round, lr, args, logger):
    total_loss = 0.0
    lr = adjust_lr(round, lr, args)

    for net_id, net in nets.items():
        net.train()
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                   lr=lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                   lr=lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                  lr=lr, momentum=args.momentum, weight_decay=args.reg)
            
        if args.alg == 'fedavg':
            loss = fedavg(net, dataloaders[net_id], optimizer, device, args)
            total_loss += loss
        elif args.alg == 'fedprox':
            loss = fedprox(net, global_model, dataloaders[net_id], optimizer, device, args)
            total_loss += loss
        elif args.alg == 'moon':
            loss = moon(net, global_model, prev_nets[net_id], dataloaders[net_id], optimizer, device, args)
            total_loss += loss

    avg_loss = total_loss / len(nets)
    logger.info(f'At round: {round}, avg_loss: {avg_loss:.4f}, lr: {lr:.6f}')
    return avg_loss, lr