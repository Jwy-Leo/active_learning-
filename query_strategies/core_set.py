import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
import torch
import gurobipy as gurobi

class CoreSet(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, tor=1e-4):
        super(CoreSet, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.tor = tor

    def mip_model(self, representation, labeled_idx, budget, delta, outlier_count, greedy_indices=None):

        model = gurobi.Model("Core Set Selection")

        # set up the variables:
        points = {}
        outliers = {}
        for i in range(representation.shape[0]):
            if i in labeled_idx:
                points[i] = model.addVar(ub=1.0, lb=1.0, vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(representation.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set the outlier budget:
        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        # build the graph and set the constraints:
        model.addConstr(sum(points[i] for i in range(representation.shape[0])) == budget, "budget")
        neighbors = {}
        graph = {}
        print("Updating Neighborhoods In MIP Model...")
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j] = [points[idx] for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j].append(outliers[j])
                model.addConstr(sum(neighbors[j]) >= 1, "coverage+outliers")

        model.__data = points, outliers
        model.Params.MIPFocus = 1
        model.params.TIME_LIMIT = 180

        return model, graph

    def get_unlabeled_idx(X_train, labeled_idx):
        """
        Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
        """
        return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]



    def query(self, n):
        GPU_VERSION = True
        dist_mat = None
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()
        print('calculate distance matrix')
        t_start = datetime.now()
        if not GPU_VERSION:
            dist_mat = np.matmul(embedding, embedding.transpose())
            sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
            dist_mat *= -2
            dist_mat += sq
            dist_mat += sq.transpose()
            dist_mat = np.sqrt(dist_mat)
        else:
            with torch.no_grad():
                embedding = torch.FloatTensor(embedding).cuda()
                dist_mat = torch.matmul(embedding, embedding.transpose(1,0))
                sq = dist_mat.diagonal().view(len(self.X), 1).clone()	# diagonal will reference
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
        print('calculate greedy solution')
        t_start = datetime.now()
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
                    gather_index = torch.LongTensor(range(q_idx)+range(q_idx+1,mat.size(0)))[:,None].repeat(1,mat.size(1))
                    mat = torch.gather(mat,0,gather_index.cuda())
                    mat = torch.cat([mat,(torch.FloatTensor(dist_mat[lb_index,int(q_idx)])[:, None]).cuda()],dim=1)
                opt = float(mat.min(dim=1)[0].max().data)

                bound_u = opt
                bound_l = opt/2.0
                delta = opt
                print(datetime.now() - t_start)

	
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        lb_flag_ = self.idxs_lb.copy()
        subset = np.where(lb_flag_==True)[0].tolist()


        eps = 0.01
        upper_bound = opt
        lower_bound = opt / 2.0
        print("Building MIP Model...")
        model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, upper_bound, outlier_count, greedy_indices=new_indices)
        model.Params.SubMIPNodes = submipnodes
        points, outliers = model.__data
        model.optimize()
        indices = [i for i in graph if points[i].X == 1]
        current_delta = upper_bound
        while upper_bound - lower_bound > eps:

            print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.getAttr(gurobi.GRB.Attr.Status) in [gurobi.GRB.INFEASIBLE, gurobi.GRB.TIME_LIMIT]:
                print("Optimization Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            else:
                print("Optimization Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0
                indices = [i for i in graph if points[i].X == 1]

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.optimize()

        return np.array(indices)



		
        # import pdb;pdb.set_trace()
        '''
		SEED = 5

		pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), n, self.n_pool), open('mip{}.pkl'.format(SEED), 'wb'), 2)

		import ipdb
		ipdb.set_trace()
		# solving MIP
		# download Gurobi software from http://www.gurobi.com/
		# sh {GUROBI_HOME}/linux64/bin/gurobi.sh < core_set_sovle_solve.py

		sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))

		if sols is None:
			q_idxs = lb_flag
		else:
			lb_flag_[sols] = True
			q_idxs = lb_flag_
		print('sum q_idxs = {}'.format(q_idxs.sum()))
		
		return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]
		'''



        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]



