import torch
import argparse
import copy
import time
import pickle

from fa import fa_utils
from utils import *
import train


def get_args():
    parser = argparse.ArgumentParser()
    ####
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', default="/data/", help="Data directory")
    parser.add_argument('--logdir', default="./logs/", help='Log directory path')
    parser.add_argument('--log_file_name', default=None, help='log file name')
    parser.add_argument('--device', default='cuda:0', help='The device to run the program')
    parser.add_argument('--num_workers', default=0, type=int, help='the number of workers for each dataloader')

    ####
    parser.add_argument('--dataset', default='cifar10', help='dataset used for training')
    parser.add_argument('--model', default='resnet18', help='neural network used in training')
    parser.add_argument('--group_norm', action='store_true', help='replace batch_norm with group_norm')
    parser.add_argument('--num_groups', type=int, default=8, help='num of groups in group_norm')

    #### Hyperparameters for training
    parser.add_argument('--optimizer', default='sgd', help='the optimizer')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size for validation or test')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scheduler', default=None, help='scheduler for rounds [linear, cosine, None]')
    parser.add_argument('--eta_min', type=float, default=0.0, help='minimum learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--reg', type=float, default=1e-3, help="L2 regularization strength")

    #### Federated learning settings
    parser.add_argument('--partition', default='noniid', help='iid, noniid')
    parser.add_argument('--alg', default='fedavg', help='communication strategy: fedavg/fedprox/fedavg_fa')
    parser.add_argument('--round', type=int, default=500, help='number of maximum communication round')
    parser.add_argument('--n_clients', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--unavailability', default='stationary', help='stationary, non-stationary, non-stationary-failure')
    parser.add_argument('--min_require_size', type=int, default=64, help='the minimum number of data for each client')
    ## Stationary
    parser.add_argument('--sample_fraction', type=float, default=0.1, help='how many clients are sampled in each round')

    ##################################################### Hypereparameters for other algs
    ## FedFA
    parser.add_argument('--fa', action='store_true')
    parser.add_argument('--pre_fa', action='store_true')
    parser.add_argument('--post_fa', action='store_true')
    parser.add_argument('--linear_scale', action='store_true')
    parser.add_argument('--sync_round', type=int, default=-1)

    ## Fedprox or MOON
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # MOON
    parser.add_argument('--use_project_head', action='store_true')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    ## FedavgM
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')

    ## FedDecorr
    parser.add_argument('--feddecorr', action='store_true')
    parser.add_argument('--feddecorr_coef', type=float, default=0.1)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device(args.device)
    logger, log_file_name = init_logger(args)

    global_train_dataset, global_val_dataset, global_test_dataset = get_global_dataset(args)
    # dataidx_map: client idx -> data idxs
    client_data_map = partition_data(global_train_dataset, args, logger)
    # print(f'cleint data map: {client_data_map}')
    clients_at_rounds = shuffle_clients(args)
    # print(f'cleints_at_rounds: {clients_at_rounds}')
    # client_datasets: list[client idx] = dataset
    client_datasets = get_client_datasets(global_train_dataset, client_data_map, args)

    # TODO: for minss
    # client_meta_datasets = get_client_meta_datasets(client_datasets, args)
    
    global_train_dataloader, global_val_dataloader, global_test_dataloader = get_global_dataloader(global_train_dataset, global_val_dataset, global_test_dataset, args)
    # client_dataloaders: list[client idx] = dataloader
    client_dataloaders = get_client_dataloaders(client_datasets, args)

    # TODO: meta learning dataloader
    # client_meta_dataloaders = get_client_meta_dataloaders(client_meta_datasets, args)

    # base_global_model: for ensuring smae intial weights
    base_global_model = init_nets(global_train_dataset, 1, args, device, True)[0]
    global_model = init_nets(global_train_dataset, 1, args, device)[0]
    base_w = base_global_model.state_dict()
    global_model.load_state_dict(base_w, strict=False)

    client_nets = init_nets(global_train_dataset, args.n_clients, args, device)
    global_w = global_model.state_dict()
    for net in client_nets.values():
        net.load_state_dict(global_w)

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    round_counter = 0
    pkl_dict = {}
    pkl_dict['args'] = vars(args)
    pkl_dict['avg_train_loss'] = []
    pkl_dict['acc'] = []
    pkl_dict['max'] = 0
    pkl_dict['avg_10'] = 0
    pkl_dict['avg_30'] = 0
    pkl_dict['avg_50'] = 0
    
    for round in range(args.round):
        logger.info(f'round:{round}')
        clients_this_round = clients_at_rounds[round]
        # print(f'clinets this round: {clients_this_round}')
        if args.pre_fa or args.post_fa:
            round_counter = fa_utils.sync_B(global_model, args, round_counter)

        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
        global_w = global_model.state_dict()
        nets_this_round = {i: client_nets[i] for i in clients_this_round}
        dataloaders_this_round = {i: client_dataloaders[i] for i in clients_this_round}
        # for MOON get get previous nets
        prev_nets = None
        if args.alg == 'moon':
            prev_nets = copy.deepcopy(nets_this_round)
            for _, net in prev_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # for fedavgM
        if args.server_momentum:
            old_w = copy.deepcopy(global_model.state_dict())

        t0 = time.time()
        avg_loss = train.train_local_net(dataloaders=dataloaders_this_round, nets=nets_this_round, global_model=global_model, prev_nets=prev_nets, device=device, round=round, args=args, logger=logger)
        t1 = time.time()
        print(f'train time: {t1 -t0}')
        pkl_dict['avg_train_loss'].append(avg_loss)
        
        # weighted mean 
        # total_data_points = sum([len(client_data_map[r]) for r in clients_this_round])
        # fed_avg_freqs = [len(client_data_map[r]) / total_data_points for r in clients_this_round]

       # weighted mean drop_last
        total_batches = sum([len(dataloaders_this_round[j]) for j in dataloaders_this_round])
        fed_avg_freqs = [len(dataloaders_this_round[j]) / total_batches for j in dataloaders_this_round ]

        # aggregation
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        # for fedavgM
        if args.server_momentum:
            delta_w = copy.deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = old_w[key] - global_w[key]
                moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                global_w[key] = old_w[key] - moment_v[key]

        global_model.load_state_dict(global_w)

        test_acc = compute_accuracy(global_model, global_test_dataloader, device=device)
        pkl_dict['acc'].append(test_acc)

        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        round_counter += 1
    

    print(args)
    pkl_dict['max'] = max(pkl_dict['acc'])
    pkl_dict['avg_10'] = avg_last_n(pkl_dict['acc'], 10)
    pkl_dict['avg_30'] = avg_last_n(pkl_dict['acc'], 30)
    pkl_dict['avg_50'] = avg_last_n(pkl_dict['acc'], 50)

    logger.info(f'max: {pkl_dict["max"]}')
    logger.info(f'acc10: {pkl_dict["avg_10"]}')
    logger.info(f'acc30: {pkl_dict["avg_30"]}')
    logger.info(f'acc50: {pkl_dict["avg_50"]}')
    with open(os.path.join(args.logdir, log_file_name + '.pkl'), 'wb') as f:
        pickle.dump(pkl_dict, f)

if __name__ == '__main__':
    main()