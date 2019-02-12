import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import torch
class KCenterGreedy(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args,X_te,Y_te):
        super(KCenterGreedy, self).__init__(X, Y, idxs_lb, net, handler, args,X_te,Y_te)

    def query(self, n):
        GPU_VERSION = True
        dist_mat = None
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        from datetime import datetime

        print('calculate distance matrix')
        t_start = datetime.now()

        if not GPU_VERSION:
                        dist_mat = np.matmul(embedding,embedding.transpose())
                        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
                        dist_mat *= -2
                        dist_mat += sq
                        dist_mat += sq.transpose()
                        dist_mat = np.sqrt(dist_mat)
        else:
                        with torch.no_grad():
                                embedding = torch.FloatTensor(embedding).cuda()
                                dist_mat = torch.matmul(embedding, embedding.transpose(1,0))
                                sq = torch.diagonal(dist_mat).view(len(self.X), 1).clone()   # diagonal will reference
                                dist_mat *= -2
                                dist_mat += sq
                                dist_mat += sq.transpose(1,0)
                                avaliable_dobule = int(float(torch.cuda.max_memory_allocated(0))/64.)
                                batch_index = int(np.sqrt(float(avaliable_dobule)))
                                for left_i in range(0,dist_mat.size(0),batch_index):
                                        right_i = min(left_i+1*batch_index,dist_mat.size(0))
                                        for left_j in range(0,dist_mat.size(1),batch_index):
                                                right_j = min(left_j+1*batch_index,dist_mat.size(1))
                                                dist_mat[left_i:right_i,left_j:right_j] = torch.sqrt(dist_mat[left_i:right_i,left_j:right_j])
                        dist_mat = dist_mat.data.cpu().numpy()

        print(datetime.now() - t_start)

        mat = dist_mat[~lb_flag, :][:, lb_flag]
		
        if not GPU_VERSION:
                        for i in range(n):
                                if i%10 == 0:
                                        print('greedy solution {}/{}'.format(i, n))
                                mat_min = mat.min(axis=1)
                                q_idx_ = mat_min.argmax()
                                q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
                                lb_flag[q_idx] = True
                                mat = np.delete(mat, q_idx_, 0)
                                mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
        else:
                        lb_index = np.where(lb_flag==False)[0]
                        mat = torch.FloatTensor(mat).cuda()
                        with torch.no_grad():
                                for i in range(n):
                                        if i%10 == 0:
                                                print('greedy solution {}/{}'.format(i, n))
                                        mat_min = mat.min(dim=1)[0]
                                        q_idx_ = mat_min.argmax()
                                        q_idx = torch.arange(self.n_pool)[lb_index][q_idx_]
                                        lb_index = set(lb_index)-set([int(q_idx)])
                                        lb_index = list(lb_index)
                                        gather_index = torch.LongTensor(list(range(q_idx))+list(range(q_idx+1,mat.size(0),1)))[:,None].repeat(1,mat.size(1))
                                        mat = torch.gather(mat,0,gather_index.cuda())
                                        mat = torch.cat([mat,(torch.FloatTensor(dist_mat[lb_index,int(q_idx)])[:, None]).cuda()],dim=1)

                                opt = float(mat.min(dim=1)[0].max().data)
                                bound_u = opt
                                bound_l = opt/2.0
                                delta = opt
                        print(datetime.now() - t_start)

        return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]
