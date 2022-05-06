
from collections import deque
import numpy as np
import torch
from torch import nn

from sklearn import mixture




class Controller(nn.Module):
	def __init__(self, encode, decode, **kwargs):
		super().__init__()
		self.encode = encode
		self.decode = decode


	def fit_latent(self, Z):
		raise NotImplementedError


	def fit(self, X):
		Z = self.encode(X)
		self.register_buffer('anchors', Z)
		self.fit_latent(Z)


	def sample_latent(self, N, gen=None):
		raise NotImplementedError


	def sample(self, N, gen=None):
		return self.decode(self.sample_latent(N, gen=gen))


	def predict(self, X):
		raise NotImplementedError



class GMM(Controller):
	def __init__(self, n_components=1, covariance_type='full', gen=None, **kwargs):
		super().__init__(**kwargs)
		self.mm = mixture.GaussianMixture(n_components=n_components,
		                                  random_state=(gen if gen is None else gen.initial_seed()),
		                                  covariance_type=covariance_type)


	def fit_latent(self, Z):
		self.mm.fit(Z.cpu().detach().numpy())


	def sample_latent(self, N, gen=None):
		z, _ = self.mm.sample(N)
		return torch.as_tensor(z, dtype=torch.float)




class LatentResponse(Controller):
	def __init__(self, nodes=None, temp=0.1, levels=3, n_neighbors=4, only_core=False,
	             neighbor_wts=None,
	             max_size=512, **kwargs):
		super().__init__(**kwargs)
		if nodes is None:
			nodes = self.NodeBuffer(max_size=max_size)
		self.nodes = nodes
		self.nodes.encode = self.encode
		self.nodes.decode = self.decode
		self.temp = temp
		self.levels = levels
		self.only_core = only_core
		if n_neighbors is None:
			assert neighbor_wts is not None
			n_neighbors = len(neighbor_wts)
		self.n_neighbors = n_neighbors
		if neighbor_wts is None:
			neighbor_wts = torch.ones(n_neighbors) / n_neighbors
		self.register_buffer('neighbor_wts', neighbor_wts)
		self.latent_dim = None


	def fit_latent(self, Z):
		self.register_buffer('anchor_responses', self.process_levels(Z))
		self.neighbor_wts = self.neighbor_wts.to(Z.device)
		self.nodes.add(Z, core=True)
		self.nodes.add(Z, core=False)
		self.latent_dim = Z.size(1)


	def process_levels(self, Z):
		Zs = [Z]
		for _ in range(self.levels):
			Zs.append(self.encode(self.decode(Zs[-1])))
		Zs = torch.stack(Zs)
		return Zs


	def generate_candidates(self, N, gen=None):
		cands = self.nodes.sample(N, gen=gen, core=self.only_core)
		noise = torch.randn(*cands.shape, device=cands.device, generator=gen)
		return cands + self.temp * noise


	def select_candidates(self, cands):
		resps = self.process_levels(cands)

		agr = torch.cdist(resps, self.anchor_responses)
		score = agr.topk(self.n_neighbors, dim=-1, largest=False)[0] @ self.neighbor_wts

		crit = score[1:].sub(score[:-1])
		gold = crit.lt(0).prod(0).bool() # strict convergence
		self.acceptance = gold.float().mean().detach()
		# gold = score[1:].sub(score[:-1]).mean(0).lt(0).bool() # average convergence
		picks = cands[gold]
		self.nodes.score(gold)
		self.nodes.add(picks)
		return picks


	def sample_latent(self, N, gen=None):
		if self.latent_dim is None:
			raise Exception('Must fit first')

		ready = []
		done = 0
		while done < N:
			cands = self.generate_candidates(N, gen=gen)
			samples = self.select_candidates(cands)
			new = min(N-done, len(samples))
			ready.append(samples[:new])
			done += len(samples)
		return torch.cat(ready, 0)


	class NodeBuffer(nn.Module):
		def __init__(self, max_size=512, **kwargs):
			super().__init__()
			self.max_size = max_size
			self.register_buffer('core', None)
			self.register_buffer('buffer', None)
			# self.inds = torch.arange(self.max_size)
			self.ptr = 0
			self.num = 0


		# def _index(self, N):
		#
		# 	if self.ptr + N > self.max_size:
		# 		inds = self.inds.narrow(0,self.ptr, N)
		# 		self.ptr += N
		# 		self.ptr %= self.max_size
		# 	else:
		# 		split = self.max_size - self.ptr
		# 		self.buffer.narrow(0, self.ptr, split).copy_(samples[:split])
		# 		self.buffer.narrow(0, 0, split).copy_(samples[split:])
		# 		self.ptr = split
		#
		# 	if N > len(self.inds[0]):
		# 		pass
		# 	elif N == len(self.inds[0]):
		# 		pass
		# 	else:
		# 		inds = self.inds[0][:N]
		# 		pass


		def add(self, samples, core=False):
			if self.buffer is None:
				self.buffer = torch.empty(self.max_size, *samples.shape[1:],
				                          device=samples.device, dtype=samples.dtype)

			if core:
				if self.core is None:
					self.core = samples.clone()
				else:
					self.core = torch.cat([self.core, samples], 0)
				return

			N = len(samples)
			if self.ptr + N > self.max_size:
				split = self.max_size - self.ptr
				self.buffer.narrow(0, self.ptr, split).copy_(samples[:split])
				self.buffer.narrow(0, 0, N-split).copy_(samples[split:])
				self.ptr = N-split
			else:
				self.buffer.narrow(0, self.ptr, N).copy_(samples)
				self.ptr += N
				self.ptr %= self.max_size
			self.num = min(self.max_size, self.num + N)


		def sample(self, N, gen=None, core=False):
			if core:
				self.sel = torch.randint(len(self.core), size=(N,), generator=gen)
				samples = self.core[self.sel]
				self.sel += self.num
				return samples

			if core is None:
				self.sel = torch.randint(self.num + len(self.core), size=(N,), generator=gen)

				samples = torch.empty(N, *self.buffer.shape[1:])

				core_sel = self.sel >= self.num
				buffer_sel = core_sel.logical_not()

				samples[core_sel] = self.core[self.sel[core_sel]-self.num]
				samples[buffer_sel] = self.buffer[self.sel[buffer_sel]]
				return samples

			self.sel = torch.randint(self.num, size=(N,), generator=gen)
			return self.buffer[self.sel]


		def score(self, scores):
			pass













