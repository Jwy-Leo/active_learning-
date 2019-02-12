import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .Training_criteria import EarlyStop as ES
class Strategy(object):
    def __init__(self, X, Y, idxs_lb, net, handler, args,test_X,test_Y):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        # self.net = net
        # self.net = torch.nn.DataParallel(net()).cuda()
        self.net = net().cuda()
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # wenyen modify
        self.optimizer = None
        self.Stop_rule = ES(3)
        self.test_data_X = test_X
        self.test_data_Y = test_Y
    def query(self, n):
        raise NotImplementedError("You should define your query rule")

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
    def _test(self,loader_te):
        self.clf.eval()
        result = []
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(loader_te):
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                result.append(pred==y)
        result = torch.cat(result,dim=0)
        performance = float(result.sum())/float(result.size(0))
        return float(performance)
    def train(self):
        
        n_epoch = self.args['n_epoch']
        # self.clf = self.net().to(self.device)
        self.clf = self.net
        # optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        # wenyen
        # if self.optimizer == None:
        #     self.optimizer = optim.Adam(self.clf.parameters(), lr=self.args['optimizer_args']['lr'])
        # optimizer = self.optimizer
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args['optimizer_args']['lr'])

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                            shuffle=True, **self.args['loader_tr_args'])
        loader_te = DataLoader(self.handler(self.test_data_X,self.test_data_Y,transform=self.args['transform']), \
                               shuffle=False, batch_size=500,num_workers = 4)
                                
        # for epoch in range(1, n_epoch+1):
        #     self._train(epoch, loader_tr, optimizer)
        self.Stop_rule._New_Update_Round()
        while True:
            self._train(0,loader_tr,optimizer)
            performance = self._test(loader_te)
            if self.Stop_rule(performance,self.clf):
                break
        self.clf.load_state_dict(self.Stop_rule.best_static_dict())
    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

