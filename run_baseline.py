import os 
import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning
#CoreSet,
from tensorboardX import SummaryWriter
import argparse

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Query mode
enum_training_strategy=["Random Query", "Shannon Entropy Sampling", "LeastConfidence Sampling", "Margin Entorpy Sampling", "KMeans Sampling", \
                       "KCenter Sampling", "CoreSet Sampling", "ALBL Sampling ", "BALDDropout(MC-dropout)", "Shannon Entorpy Sampling with dropout", \
                       "LeastConfidence Sampling with dropout", "Margin Entropy Sampling with dropout", "AdversarialBIM", "Deepfool"]

# Datasets
datasets_setting = {'MNIST':
                {'n_epoch': 200, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 10},
                 'loader_te_args':{'batch_size': 10000, 'num_workers': 10},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
}

    
def arguments():

    parse = argparse.ArgumentParser( description = "Active learning baselines", formatter_class = argparse.RawTextHelpFormatter)

    # Query mode selection
    comment = "".join(["{0:2d} : {1}(default)\n".format(index, name) if index==0 else "{0:2d} : {1}\n".format(index,name) for index,name in enumerate(enum_training_strategy)])
    parse.add_argument("--query_mode", type = int, default = 0, help = comment)
    # Datasets selection
    comment = "".join(["{}(default)\n".format(name) if index==0 else "{}\n".format(name) for index,name in enumerate(datasets_setting.keys())])    
    parse.add_argument("--datasets", type = str, default = "MNIST", help = comment)
    # Log setting
    parse.add_argument("--tb_log", type = str, default = "exp/", help = "tensorboard log path\t\t(default:'exp/')")

    # Util setting
    parse.add_argument("--seed", type = int, default = 0, help = "random seed\t\t\t(default:0)")
    parse.add_argument("--Init_N", type = int, default = 50, help = "Number of initial image set\t(default:50)")
    parse.add_argument("--NIQ", type = int, default = 1, help = "Number of images/query\t\t(default:1)")
    parse.add_argument("--qt", type = int, default = 50, help = "query times\t\t\t(default:50)")

    
    args = parse.parse_args()
    print(args)

    return args

def main(args):
    # Set log 
    writer = SummaryWriter(os.path.join(args.tb_log,enum_training_strategy[args.query_mode]))

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load datasets
    X_tr, Y_tr, X_te, Y_te, X_eval, Y_eval = load_datasets(args)
    
    # generate initial labeled pool
    idxs_lb = np.zeros(X_tr.shape[0], dtype=bool)
    idxs_tmp = np.arange(X_tr.shape[0])
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:args.Init_N]] = True

    # Load Network
    net = get_net(args.datasets)

    # Get dataloader
    handler = get_handler(args.datasets)

    # Training hyperparameter
    H_param = datasets_setting[args.datasets]

    if args.query_mode == 0 :
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
    elif args.query_mode == 1 :
        strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
    elif args.query_mode == 2 :
        strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
    elif args.query_mode == 3 :
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
    elif args.query_mode == 4 :
        strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
    elif args.query_mode == 5 :
        strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
        #raise NotImplementedError("gpu memory issue in line 73 of kcenter file")
    elif args.query_mode == 6 :
        # strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)
        raise NotImplementedError("Yet to fix coreset")
        pass
    elif args.query_mode == 7 :
        albl_list = [EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te),KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te)]
        strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, H_param, strategy_list = albl_list, X_te = X_te, Y_te = Y_te, delta=0.1)
    elif args.query_mode == 8 :
        strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, n_drop=10)
    elif args.query_mode == 9 :
        strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, n_drop=10)
    elif args.query_mode == 10 :
        strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, n_drop=10)
    elif args.query_mode == 11 :
        strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, n_drop=10)
    elif args.query_mode == 12 :
        strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, eps=0.05)
    elif args.query_mode == 13 :
        strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, H_param, X_te, Y_te, max_iter=50)
    else: 
        raise NotImplementedError("We don't have the index query strategy")

    # print info
    print(args.datasets)
    print('SEED {}'.format(args.seed))
    print(type(strategy).__name__)

    # round 0 accuracy
    strategy.train()
    P = strategy.predict(X_eval, Y_eval)
    acc = np.zeros(args.qt+1)
    acc[0] = 1.0 * (Y_eval==P).sum().item() / len(Y_eval)
    print('Round 0\ntesting accuracy {}'.format(acc[0]))

    for rd in range(1, args.qt+1):
        print('Round {}'.format(rd))

        # query
        q_idxs = strategy.query(args.NIQ)
        idxs_lb[q_idxs] = True

        # update
        strategy.update(idxs_lb)
        strategy.train()

        # round accuracy
        P = strategy.predict(X_eval, Y_eval)
        acc[rd] = 1.0 * (Y_eval==P).sum().item() / len(Y_eval)
        writer.add_scalar('test_ac',acc[rd],rd)
        print('testing accuracy {}'.format(acc[rd]))

    # print results
    print('SEED {}'.format(args.seed))
    print(type(strategy).__name__)
    print(acc)

def load_datasets(args):
    
    X_tr, Y_tr, X_eval, Y_eval = get_dataset(args.datasets)
    #TIN = int(float(X_tr.shape[0])*0.8)
    TIN = 100
    
    # Testing data split 
    index_te = np.array([np.random.choice(np.where(Y_tr.data.numpy()==i)[0],int(float(len(Y_tr))*0.2*0.1),replace=False) for i in range(10)]).reshape(-1)
    X_te = X_tr[index_te,...]
    Y_te = Y_tr[index_te,...]
    index_train = np.array(list(set(list(range(len(Y_tr))))-set(index_te.tolist())))
    X_tr = np.take(X_tr,index_train,axis=0)
    Y_tr = np.take(Y_tr,index_train,axis=0)

    X_tr = X_tr[:TIN]
    Y_tr = Y_tr[:TIN]
    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te) 
    n_eval = len(Y_eval)
    print('number of labeled pool: {}'.format(args.Init_N))
    print('number of unlabeled pool: {}'.format(n_pool - args.Init_N))
    print('number of test pool : {} '.format(n_test))
    print('number of evaluation pool: {}'.format(n_eval))
    return X_tr, Y_tr, X_te, Y_te, X_eval, Y_eval

if __name__=="__main__":
    args = arguments()
    main(args)
