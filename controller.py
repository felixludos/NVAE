
import numpy as np
import torch
from torch import nn

from sklearn import mixture




class Controller(nn.Module):
	def __init__(self, encode, decode, **kwargs):
		super().__init__()
		self.encode = encode
		self.decode = decode


	def fit(self, X):
		raise NotImplementedError


	def sample(self, N):
		raise NotImplementedError


	def predict(self, X):
		raise NotImplementedError



class GMM(Controller):
	def __init__(self, n_components=1, covariance_type='full', gen=None, **kwargs):
		super().__init__(**kwargs)
		self.mm = mixture.GaussianMixture(n_components=n_components,
		                                  random_state=(gen if gen is None else gen.initial_seed()),
		                                  covariance_type=covariance_type)


	def fit(self, X):
		Z = self.encode(X)
		self.register_buffer('anchors', Z)
		self.mm.fit(Z.cpu().detach().numpy())



class LatentResponse(Controller):
	def __init__(self, nodes=None, temp=0.1, levels=3, n_neighbors=4,
	             max_size=256, **kwargs):
		super().__init__(**kwargs)
		if nodes is None:
			nodes = self.NodeBuffer(max_size=max_size)
		self.nodes = nodes
		self.temp = temp


	def fit(self, X):
		Z = self.encode(X)
		self.register_buffer('anchors', Z)





	def sample(self, N, gen=None):
		pass



	class NodeBuffer(nn.Module):
		def __init__(self, max_size=256, **kwargs):
			super().__init__()
			self.max_size = max_size
















