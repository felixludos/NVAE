import sys, os
from pathlib import Path
from tqdm import tqdm
from omnibelt import unspecified_argument, agnosticmethod, get_now
import omnifig as fig

import numpy as np
import torch
from torch import nn
from torch import optim as opt
from torch import distributions as distrib
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from plethora.framework.random import Sampler


from model import AutoEncoder, Normal
import utils
# import datasets

from controller import LatentResponse, GMM
from train_small import train_rep

import omnilearn
from omnilearn import util

from plethora import datasets, tasks, framework as fm
from plethora.framework import export, load_export

# _p = os.getenv('PLETHORA_PATH')
# print(f'loaded conditional_eval {_p} {fm.Rooted._DEFAULT_MASTER_ROOT}')


class ModelWrapper(fm.Encoder, fm.Decoder, fm.Sampler):
	def __init__(self, base, **kwargs):
		super().__init__(din=base.din, dout=base.dout)
		self.base = base
		self.latent_dim = self.base.latent_dim


	def encode(self, observation):
		with torch.no_grad():
			latent = self.base.encode(observation)
			if isinstance(latent, distrib.Distribution):
				latent = latent.mean
		return latent


	def decode(self, latent):
		with torch.no_grad():
			reconstruction = self.base.decode(latent)
		return reconstruction


	def sample_latent(self, *shape, gen=None):
		prior = torch.randn(*shape, self.latent_dim, generator=gen)
		return prior.to(self.base.device)


	def _sample(self, shape, gen):
		return self.decode(self.sample_latent(*shape, gen=gen))



@fig.AutoComponent('old-model')
def simple_model(run_name, root=None, device=None):
	fig.initialize('sae')

	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if root is None:
		root = os.getenv('OMNILEARN_SAVE_DIR', '.')
	root = Path(root)
	load_config = fig.get_config(path=run_name, root=str(root), **{'override.device': device})
	load_config.set_silent(True)
	run = fig.run('load-run', load_config)
	# A = run.get_config()
	# A.set_silent(True)
	# info = A.pull('info')

	model = run.get_model()
	model.switch_to('val')
	model.to(device);
	return ModelWrapper(model)



def load_dataset(dname, mode='train'):
	if dname == 'mnist':
		dataset = datasets.MNIST(batch_size=256, mode=mode).prepare()
	elif dname == 'cifar10':
		dataset = datasets.CIFAR10(batch_size=128, mode=mode).prepare()
	elif 'celeba' in dname:
		dataset = datasets.CelebA(resize=64 if '64' in dname else None, batch_size=64, mode=mode).prepare()
	else:
		assert False
	return dataset



def load_encoder(A, enc_name, dname, cycles=0, device='cuda'):
	if enc_name == 'flat':
		encoder = NVAE_Flat(dname=dname, num_dim=None, threshold=None, device=device, auto_cpu=False)
	else:
		if enc_name is None:
			encoder = None
		else:
			encoder = simple_model(enc_name, root=A.pull('model-root', None))
	if encoder is not None and cycles > 0:
		encoder = ResponseModel(encoder, cycles=cycles)
		print('Using response model')
	return encoder



@fig.Script('task')
def simple_task(A):
	device = A.pull('device', 'cuda' if torch.cuda.is_available() else 'cpu')
	pbar = tqdm if A.pull('pbar', False) else None
	extra = A.pull('extra', '')

	dname = A.pull('dataset', 'mnist')

	enc_name = A.pull('encoder', 'flat')
	cycles = A.pull('cycles', 0)

	task_name = A.pull('task', 'fid')

	root = A.pull('root', str(fm.Rooted.get_root()/'tasks'))
	root = Path(root)
	if not root.exists():
		os.makedirs(str(root))
	
	mode = A.pull('mode', 'train')
	mode_ident = A.pull('mode-ident', mode)

	dataset = load_dataset(dname, mode=mode)

	model = load_encoder(A, enc_name, dname, cycles=cycles, device=device)

	seed = A.pull('seed', 67280421310721)

	
	other_kwargs = A.pull('task_kwargs', {})
	
	num_samples = A.pull('num-samples', '<>sample-limit', 10000)
	
	fid_dim = A.pull('inception-dim', 2048)
	if task_name == 'fid':
		task_name = f'{task_name}{fid_dim}'
		task = tasks.FID_GenerationTask(dataset=dataset, pbar=pbar, sample_limit=num_samples,
		                                generator=model, inception_dim=fid_dim, **other_kwargs)
	elif task_name == 'pr':
		task_name = f'{task_name}{fid_dim}'
		task = tasks.PR_GenerationTask(dataset=dataset, pbar=pbar, sample_limit=num_samples,
		                                generator=model, inception_dim=fid_dim, **other_kwargs)
	elif task_name == 'inf':
		task = tasks.InferenceTask(dataset=dataset, pbar=pbar, sample_limit=num_samples,
		                           encoder=model, **other_kwargs)
	else:
		assert False, f'{task_name}'

	run_name = f'{task_name}_{dname}_{enc_name}'
	if cycles > 0:
		run_name += f'r{cycles}'
	if len(extra):
		run_name = f'{run_name}_{extra}'
	run_name = A.pull('name', run_name)

	path = root / run_name
	path.mkdir(exist_ok=True)
	
	with fm.using_rng(seed=seed):
		with torch.no_grad():
			info = task.compute()

		if 'score' in info:
			score = info['score']
			print(f'Score {run_name}: {score}')

		info['model_name'] = enc_name
		info['dataset_fingerprint'] = dataset.fingerprint()
		info['dataset_name'] = dname
		info['seed'] = seed

	if not A.pull('include-heavy', False):
		for heavy in task.heavy_results():
			del info[heavy]
	
	if A.pull('publish-scores', True):
		writer = SummaryWriter(str(path))
		for key in task.score_names():
			writer.add_scalar(f'{key}/{mode_ident}', info[key], global_step=0)
		if 'score' in info:
			writer.add_scalar(f'score-{task_name}/{mode_ident}', info['score'], global_step=True)
		writer.close()

	return export(info, name=f'info_{mode_ident}', root=path)



@fig.Script('strain')
def linear_optimize(A):
	device = A.pull('device', 'cuda' if torch.cuda.is_available() else 'cpu')
	pbar = tqdm if A.pull('pbar', False) else None
	extra = A.pull('extra', '')

	dname = A.pull('dataset', 'mnist')

	enc_name = A.pull('encoder', 'flat')
	cycles = A.pull('cycles', 0)

	root = A.pull('root', str(fm.Rooted.get_root()/'runs'))
	root = Path(root)
	if not root.exists():
		os.makedirs(str(root))

	run_name = f'{dname}_{enc_name}'
	if cycles > 0:
		run_name += f'r{cycles}'
	if len(extra):
		run_name = f'{run_name}_{extra}'

	run_name = A.pull('name', run_name)
	run_name = f'{run_name}_{get_now()}'
	offset = len(list(root.glob(f'{run_name}*')))
	if offset > 0:
		run_name += f'_{offset}'
	path = root / run_name
	path.mkdir()
	writer = SummaryWriter(str(path))

	dataset = load_dataset(dname)
	dout = dataset.dout.shape[0]

	if 'celeba' in dname:
		targets = dataset.get_target()
		sup = targets.sum(0)
		wt = (len(targets) - sup) / sup
		criterion = nn.BCEWithLogitsLoss(pos_weight=wt.to(device))
	else:
		criterion = nn.CrossEntropyLoss()

	encoder = load_encoder(A, enc_name, dname, cycles=cycles, device=device)
	din = dataset.observation_space.shape.numel() if encoder is None else encoder.latent_dim

	model_config = A.pull('model', None, raw=True)
	if model_config is None:
		model = nn.Linear(din, dout)
	else:
		model_config.push('din', din)
		model_config.push('dout', dout)
		model = model_config.pull_self()

	print(model)
	print(f'Number of parameters: {util.count_parameters(model)}')
	writer.add_text('model-str', str(model))

	lr = A.pull('lr', 0.001)
	l2 = A.pull('l2', 0.0001)
	optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=l2)

	model.to(device)
	model.train()
	# optimizer.to(device)

	def process_batch(batch):
		X, Y = batch['observation'].to(device), batch['target'].to(device)
		if encoder is None:
			Z = X.view(X.size(0), -1)
		else:
			with torch.no_grad():
				Z = encoder.encode(X)
		return Z, Y

	batch_size = A.pull('batch-size', batch_size)
	val_step = A.pull('val-step', 10)
	train_step = A.pull('train-step', 10)
	ckpt_step = A.pull('ckpt-step', 100)
	eta = 1 / A.pull('stat-eta', train_step)

	seed = A.pull('seed', 67280421310721)

	def checkpoint(i, **stats):
		export({
			'seed': seed,
			'iteration': i,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			**stats,
		}, name=f'checkpoint{i}', root=path)

	def validate(i, key='loss/val', full_epoch=False):
		with torch.no_grad():
			val_losses = []
			for batch in valset.get_iterator(epoch=1, batch_size=batch_size):
				X, Y = process_batch(batch)
				y = model(X)
				valloss = criterion(y, Y).item()
				val_losses.append(valloss)
				if not full_epoch:
					break
		valloss = torch.as_tensor(val_losses).mean().item()
		writer.add_scalar(key, valloss, i)
		return valloss


	# loss = torch.as_tensor(float('nan'))
	trainloss = None
	valloss = None


	with fm.using_rng(seed=seed):
		trainset, valset = dataset.split([None, 0.1], shuffle=True)

		for i, batch in enumerate(trainset.get_iterator(epochs=A.pull('epochs',2), num_batches=A.pull('budget', None),
		                                                shuffle=True, batch_size=batch_size,
		                                                pbar=pbar, pbar_samples=False)):

			if i % val_step == 0:
				valloss = validate(i)
			if i>0 and i % ckpt_step == 0:
				checkpoint(i, val_loss=valloss, train_loss=trainloss)

			optimizer.zero_grad()

			X, Y = process_batch(batch)
			y = model(X)
			loss = criterion(y, Y)

			loss.backward()
			optimizer.step()

			trainloss = loss.item() if trainloss is None else (eta*loss.item() + (1-eta)*trainloss)
			status = f'train={trainloss:2.3f}, val={valloss:2.3f}'
			batch.progress.set_description(status)
			if i % train_step == 0:
				writer.add_scalar('loss/train', trainloss, i)
				writer.flush()
				if pbar is None:
					t = get_now('%H:%M:%S')
					print(f'{t} - it: {i} - {status}')

		valloss = validate(i+1, key='loss/val-end', full_epoch=True)
		checkpoint(i+1, val_loss=valloss, train_loss=trainloss)

	writer.close()
	return valloss



# @fig.Script('cgen')
# def conditional_generation(A):
#
#
# 	print('worked')
#
# 	pass



class NVAE_Wrapper(nn.Module, Sampler, fm.abstract.Encoder, fm.abstract.Decoder):
	def __init__(self, dname=None, root=None, device='cuda', base_prior=False, auto_cpu=True):
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
		self.auto_cpu = auto_cpu
		self.full_dims = self._full_dims_by_dataset.get(dname)
		self.dname = dname
		self.device = device
		self.ckpt_path = path
		self._global_prior_noise = self.sample_base_prior(gen=torch.Generator().manual_seed(67280421310721))
		if not base_prior:
			self._global_prior_noise = [p*0 for p in self._global_prior_noise]


	_full_dims_by_dataset = {
		'mnist': [(20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 4, 4), (20, 8, 8), (20, 8, 8), (20, 8, 8),
		          (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8)],
		# 'cifar10': [],
		# 'ffhq': [],
		'celeba_64': [(20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 8, 8), (20, 16, 16), (20, 16, 16), (20, 16, 16),
		         (20, 16, 16), (20, 16, 16), (20, 16, 16), (20, 16, 16), (20, 16, 16), (20, 16, 16), (20, 16, 16),
		         (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32),
		         (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32),
		         (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32), (20, 32, 32)],
	}


	def sample_base_prior(self, N=1, t=1., gen=None):
		if gen is None:
			gen = self.gen
		return [torch.randn(N, *shape, generator=gen).mul(t).to(self.device) for shape in self.full_dims]


	def sample_prior(self, N=1, gen=None):
		return self.sample_base_prior(N, gen=gen)


	def _sample(self, shape, gen):
		assert len(shape) == 1, f'bad shape: {shape}'
		prior = self.sample_prior(*shape, gen=gen)
		return self.decode(prior)


	def encode(self, x, t=1., prior=False, distr=True, eps_table=unspecified_argument, **kwargs):
		if eps_table is unspecified_argument:
			eps_table = self._global_prior_noise
		with torch.no_grad():
			x = x.to(self.device)
			return self.model.encode(x, t=t, distr=distr, prior=True, eps_table=eps_table, **kwargs)


	def decode(self, zs, t=1., eps_table=unspecified_argument, **kwargs):
		if eps_table is unspecified_argument:
			eps_table = self._global_prior_noise
		with torch.no_grad():
			out = self.model.decode(zs, t=t, eps_table=eps_table, **kwargs)
			out = self.to_img(out)
			if self.auto_cpu:
				return out.cpu()
			return out


	def to_img(self, logits):
		with torch.no_grad():
			img = self.model.decoder_output(logits)
			img = img.mean if isinstance(img, torch.distributions.bernoulli.Bernoulli) \
				else img.sample()
		return img#.cpu()

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


	def sample_prior(self, N=1, gen=None):
		return torch.randn(N, self.latent_dim, generator=gen)


	def calc_dim_spec(self, X, num_dim=10, threshold=None):
		# print('calculating dim spec')
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
			if self.dim_spec is None and (self.num_dim is not None or self.threshold is not None):
				self.dim_spec = self.calc_dim_spec(x, num_dim=self.num_dim, threshold=self.threshold)
			dims = self.dim_spec

		B = x.size(0)
		zs = super().encode(x)
		if dims is not None:
			out = torch.cat([self.dim_sel(z, ds) for z, ds in zip(zs, dims)], -1)
		else:
			out = torch.cat([z.view(B,-1) for z in zs], -1)
		if self.auto_cpu:
			return out.cpu()
		return out


	def decode(self, z, dims=None):
		if dims is None:
			dims = self.dim_spec
		vals = z
		zs = [z.expand(len(vals), *z.shape[1:]).contiguous() for z in self._global_prior_noise]
		i = 0
		vals = vals.to(self.device)
		if dims is None:
			for z in zs:
				numel = z.shape[1:].numel()
				z.view(len(vals), -1).copy_(vals.narrow(1, i, numel))
				i += numel
		else:
			for z, dim in zip(zs, dims):
				self.dim_set(z, vals.narrow(1, i, len(dim)), dim)
				i += len(dim)
		return super().decode(zs)



	pass



class ResponseModel(fm.abstract.Generator, fm.abstract.Encoder, fm.abstract.Decoder):
	def __init__(self, base, cycles=1, **kwargs):
		super().__init__(**kwargs)
		self.base = base
		self.cycles = cycles

	def get_response(self, z, n=None):
		if n is None:
			n = self.cycles
		for _ in range(n):
			z = self.base.encode(self.base.decode(z))
		return z


	def encode(self, x):
		return self.get_response(self.base.encode(x))


	def decode(self, z):
		return self.base.decode(self.get_response(z))


	def sample_latent(self, *shape, gen=None):
		return self.get_response(self.base.sample_latent(*shape, gen=gen))


	def sample(self, *shape, gen=None):
		return self.base.decode(self.get_response(self.base.encode(self.base.sample(*shape, gen=gen)), n=self.cycles-1))






















