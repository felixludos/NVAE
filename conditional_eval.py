import sys, os
from pathlib import Path
from omnibelt import unspecified_argument, agnosticmethod
import omnifig as fig

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from plethora.framework.random import Generator

from model import AutoEncoder, Normal
import utils
# import datasets

from controller import LatentResponse, GMM


class NVAE_Wrapper(nn.Module, Generator):
	def __init__(self, dname=None, root=None, device='cuda'):
		super().__init__()
		# dname = 'mnist'
		if root is None:
			root = 'checkpoints'
		root = Path(root)
		path = root / dname
		if dname == 'cifar10':
			path = path / 'qualitative'
		path = path / 'checkpoint.pt'

		ckpt = torch.load(path, map_location='cpu')

		# checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
		args = ckpt['args']
		if not hasattr(args, 'ada_groups'):
			args.ada_groups = False
		if not hasattr(args, 'min_groups_per_scale'):
			args.min_groups_per_scale = 1
		if not hasattr(args, 'num_mixture_dec'):
			args.num_mixture_dec = 10
		# if eval_args.batch_size > 0:
		#     args.batch_size = eval_args.batch_size
		arch_instance = utils.get_arch_cells(args.arch_instance)

		model = AutoEncoder(args, None, arch_instance)
		model.load_state_dict(ckpt['state_dict'], strict=False);
		model.to(device)
		# model = model.cuda();

		model.eval();

		self.args = args
		self.model = model
		self.full_dims = self._full_dims_by_dataset.get(dname)
		self.dname = dname
		self.device = device
		self.ckpt_path = path
		self._global_prior_noise = self.sample_prior(gen=torch.Generator().manual_seed(67280421310721))


	_full_dims_by_dataset = {
		'mnist': [(20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 8, 8), (20, 8, 8), (20, 8, 8),
		          (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8)],
		# 'cifar10': [],
		# 'celeba': [],
		# 'ffhq': [],
	}


	def sample_prior(self, N=1, t=1., gen=None):
		if gen is None:
			gen = self.gen
		return [torch.randn(N, *shape, generator=gen).mul(t).to(self.device) for shape in self.full_dims]


	def encode(self, x, t=1., prior=False, distr=True, eps_table=unspecified_argument, **kwargs):
		if eps_table is unspecified_argument:
			eps_table = self._global_prior_noise
		with torch.no_grad():
			x = x.cuda()
			return self.model.encode(x, t=t, distr=distr, prior=True, eps_table=eps_table, **kwargs)


	def decode(self, zs, t=1., eps_table=unspecified_argument, **kwargs):
		if eps_table is unspecified_argument:
			eps_table = self._global_prior_noise
		with torch.no_grad():
			out = self.model.decode(zs, t=t, eps_table=eps_table, **kwargs)
			return self.to_img(out).cpu()


	def to_img(self, logits):
		with torch.no_grad():
			img = self.model.decoder_output(logits)
			img = img.mean if isinstance(img, torch.distributions.bernoulli.Bernoulli) \
				else img.sample()
		return img.cpu()

	pass



class NVAE_Native(NVAE_Wrapper):
	def __init__(self, temperature=1., **kwargs):
		super().__init__(**kwargs)
		self.temperature = temperature


	def encode(self, x):
		return super().encode(x, t=self.temperature, prior=False, distr=True, eps_table=None)


	def decode(self, zs):
		return super().decode(zs, t=self.temperature, eps_table=None)



class NVAE_Simple(NVAE_Wrapper):
	def encode(self, x, t=1.):
		zs, ps = super().encode(x, t=1., prior=True, distr=False)
		return [z.sub(p.mu).div(p.sigma) for z, p in zip(zs, ps)]


	def decode(self, zs, t=1.):
		return super().decode([None], t=1., eps_table=zs)



class NVAE_Flat(NVAE_Simple):
	def __init__(self, dim_spec=None, num_dim=10, threshold=None, **kwargs):
		super().__init__(**kwargs)
		self.threshold = threshold
		self.num_dim = num_dim

		self.dim_spec = dim_spec

	@property
	def dim_spec(self):
		return self._dim_spec
	@dim_spec.setter
	def dim_spec(self, spec):
		if spec is not None:
			self.latent_dim = sum(len(dim) for dim in spec)
			self._dim_spec = [torch.as_tensor(dim).to(self.device).long() for dim in spec]
		else:
			self.latent_dim = sum(torch.Size(v).numel() for v in self.full_dims)
			self._dim_spec = None


	def calc_dim_spec(self, X, num_dim=10, threshold=None):
		Z, P = super(NVAE_Simple, self).encode(X, prior=True, distr=True)
		B = Z[0].mu.size(0)

		vrs = torch.as_tensor([z.mu[0].numel() for z in Z]).to(Z[0].mu).long()
		rungs = torch.cat([torch.ones(v.item())*i for i, v in enumerate(vrs)]).to(vrs)
		inds = torch.cat([torch.arange(v) for v in vrs]).to(vrs)
		informatives = torch.cat([z.sigma.div(p.sigma).view(B, -1) for z,p in zip(Z, P)], -1)
		# informatives = informatives.min(0)[0]
		informatives = informatives.mean(0)

		if threshold is None:
			assert num_dim is not None
			picks = informatives.topk(num_dim, largest=False)[1]
			sel = torch.zeros_like(informatives)
			sel[picks] = 1.
			sel = sel.bool()
			picks = sel
		else:
			picks = informatives.lt(threshold)

		dim_spec = [inds[picks * (rungs == i)].tolist() for i in range(len(Z))]
		return dim_spec


	@staticmethod
	def dim_sel(t, ds):
		B = t.size(0)
		return t.view(B, -1).index_select(1, ds)


	@staticmethod
	def dim_set(t, v, ds):
		B = t.size(0)
		v = torch.as_tensor(v).expand(*v.shape).to(t)
		t.view(B, -1).index_copy_(1, ds, v)


	def encode(self, x, dims=None):
		if dims is None:
			if self.dim_spec is None and self.num_dim is not None or self.threshold is not None:
				self.dim_spec = self.calc_dim_spec(x, num_dim=self.num_dim, threshold=self.threshold)
			dims = self.dim_spec

		B = x.size(0)
		zs = super().encode(x)
		if dims is not None:
			return torch.cat([self.dim_sel(z, ds) for z, ds in zip(zs, dims)], -1).cpu()
		return torch.cat([z.view(B,-1)], -1).cpu()


	def decode(self, z, dims=None):
		if dims is None:
			dims = self.dim_spec
		vals = z
		zs = [z.expand(len(vals), *z.shape[1:]).contiguous() for z in self._global_prior_noise]
		i = 0
		vals = vals.to(self.device)
		for z, dim in zip(zs, dims):
			self.dim_set(z, vals.narrow(1, i, len(dim)), dim)
			i += len(dim)
		return super().decode(zs)



	pass



@fig.Script('cgen')
def conditional_generation(A):




	pass


























