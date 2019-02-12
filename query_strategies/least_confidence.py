import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args,X_te,Y_te):
		super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args,X_te,Y_te)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		U = probs.max(1)[0]
		return idxs_unlabeled[U.sort()[1][:n]]
