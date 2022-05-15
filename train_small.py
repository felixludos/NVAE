import sys, os
from pathlib import Path
from tqdm import tqdm
from omnibelt import unspecified_argument, agnosticmethod, get_now
import omnifig as fig

import numpy as np
import torch
from torch import nn
from torch import optim as opt
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from plethora.framework.random import Sampler


from model import AutoEncoder, Normal
import utils
# import datasets

from controller import LatentResponse, GMM

import omnilearn
from omnilearn import util

from plethora import datasets, tasks, framework as fm
from plethora.framework import export, load_export


class Encoder(fm.Encoder):
	pass


class Decoder(fm.Decoder):
	pass

import time
from torch.utils.tensorboard import SummaryWriter


@fig.AutoComponent('logger')
class Logger(object):
	def __init__(self, log_dir=None, **kwargs):
		self.tblog = SummaryWriter(log_dir=log_dir, **kwargs)
		self.global_step = None
		self.tag_fmt = None

	def add_hparams(self, param_dict, metrics={}, run_name=None):
		# metrics = {key if self.tag_fmt is None else self.tag_fmt.format(key): val for key, val in metrics.items()}
		self.tblog.add_hparams(param_dict, metrics, run_name=run_name)

	def add(self, data_type, tag, *args, global_step=None, **kwargs):  # TODO: test and maybe clean
		if self.tblog is None:
			return None

		add_fn = self.tblog.__getattribute__('add_{}'.format(data_type))

		if global_step is None:
			global_step = self.global_step

		if self.tag_fmt is not None:
			tag = self.tag_fmt.format(tag)

		add_fn(tag, *args, global_step=global_step, **kwargs)


	def flush(self):
		self.tblog.flush()


	def close(self):
		self.tblog.close()

from tqdm import tqdm

class LogTqdm(tqdm):
	def __init__(self, *args, bar_format='{l_bar}{r_bar}', disable=True, **kwargs):
		super().__init__(*args, bar_format=bar_format, disable=disable, **kwargs)




class Trainer(fm.Trainer):
	def __init__(self, model=None, ckpt_freq=None, val_freq=None, monitor_freq=None, **kwargs):
		super().__init__(model=model, **kwargs)
		self.ckpt_freq =  ckpt_freq
		self.val_freq = val_freq
		self.monitor_freq = monitor_freq

	def checkpoint(self):
		pass


	def validate(self):
		pass


	def monitor(self):
		pass




	pass




@fig.Script('train')
def train_rep(A):



	pass


def train_rep(A):

	device = A.pull('device', 'cuda' if torch.cuda.is_available() else 'cpu')
	pbar = tqdm if A.pull('pbar', False) else None
	extra = A.pull('extra', '')

	dname = A.pull('dataset', 'mnist')

	# region Run Files

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

	#endregion

	# region Dataset

	if dname == 'mnist':
		dataset = datasets.MNIST().prepare()
		criterion = nn.CrossEntropyLoss()
		batch_size = 512
	elif dname == 'cifar10':
		dataset = datasets.CIFAR10().prepare()
		criterion = nn.CrossEntropyLoss()
		batch_size = 512
	elif dname == 'celeba_64':
		dataset = datasets.CelebA(resize=64 if '64' in dname else None).prepare()

		targets = dataset.get_target()
		sup = targets.sum(0)

		wt = (len(targets) - sup) / sup
		criterion = nn.BCEWithLogitsLoss(pos_weight=wt.to(device))
		batch_size = 256
	else:
		assert False

	#endregion

	A.push('obs_shape', dataset.observation_space.shape)
	model = A.pull('model')

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
		X = batch['observation'].to(device)
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
			batch.set_description(status)
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


	pass


