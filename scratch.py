import sys, os, time, shutil, random
from pathlib import Path
# _cwd = os.getcwd()
# os.chdir(Path(_cwd)/'..')
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:98% !important; }</style>"))
# %load_ext autoreload
# %autoreload 2
# %pdb
from yaml import dump
import omnifig as fig
import numpy as np
np.set_printoptions(linewidth=120)
import pickle
from tabulate import tabulate
from tqdm import tqdm_notebook as tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from omnibelt import unspecified_argument
# import torchvision.models
from torch.utils.data import Dataset, DataLoader, TensorDataset
# import timm
torch.set_printoptions(linewidth=300, sci_mode=False)
# %matplotlib notebook
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import animation
import matplotlib as mpl
# mpl.rc('image', cmap='gray')
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import normalize

# import gpumap
from c3linearize import linearize, class_graph

# fig.initialize()

import omnilearn as learn
from omnilearn import models
from omnilearn import util
from omnilearn.data import InterventionSamplerBase

from sklearn.decomposition import PCA
import sklearn.datasets


# from src import sample_full_interventions, response_mat, factor_reponses
# from src.responses import sample_full_interventions

# fig.initialize('sae')
# from omnifig.projects import sae
# import networkx as nx

# # from src import sample_full_interventions, response_mat, factor_reponses
# dataset = None
# src = None

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
distinct_colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",]
rank = 0
world_size = 1
init_method="file:///C:/Users/anwan/Documents/tmp/env"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)

from model import AutoEncoder
import utils
import datasets


imgroot = Path('figures')
imgroot.mkdir(exist_ok=True)


dname = 'celeba_64'
dname = 'mnist'
# dname = 'cifar10'


root = Path('checkpoints')
ds_ckpts = list(root.glob('*'))
ds_ckpts

path = random.choice(ds_ckpts)
# path = root / 'mnist'
path = root / dname
if dname == 'cifar10':
    path = path / 'qualitative'
path



ckpt = torch.load(path/'checkpoint.pt', map_location='cpu')

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
model = model.cuda();

model.eval();


def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.sample(num_samples, t)
        model.eval()

temperature = 0.7
bn_eval_mode = True



_global_eps_table = None
def encode(x, t=1., prior=False, distr=True, eps_table=unspecified_argument, **kwargs):
    if eps_table is unspecified_argument:
        eps_table = _global_eps_table
    with torch.no_grad():
        x = x.cuda()
        return model.encode(x, t=t, distr=distr, prior=prior,eps_table=eps_table,  **kwargs)

def decode(all_q, t=1., eps_table=unspecified_argument, **kwargs):
    if eps_table is unspecified_argument:
        eps_table = _global_eps_table
    with torch.no_grad():
#         all_q = [q.cuda() for q in all_q]
        out = model.decode(all_q, t=t,eps_table=eps_table,  **kwargs)
        return to_img(out).cpu()

def to_img(logits):
    with torch.no_grad():
        img = model.decoder_output(logits)
        img = img.mean if isinstance(img, torch.distributions.bernoulli.Bernoulli) \
                            else img.sample()
    return img.cpu()

def enc(x):
    zs, ps = encode(x, distr=False, prior=True, eps_table=None)
    return [z.sub(p.mu).div(p.sigma) for z, p in zip(zs, ps)]
    # return [out[0].mu] + [o.eps for o in out[1:]]
def dec(q):
    return decode([None], eps_table=q)



if dname == 'mnist':
    bsnum = 24
    dataset = fig.quick_run('load-data', name='mnist', batch_size=bsnum, seed=11)
elif dname == 'cifar10':
    bsnum = 24
    dataset = fig.quick_run('load-data', name='cifar10', batch_size=bsnum, seed=11)
elif dname == 'celeba_64':
    bsnum = 12
    dataset = fig.quick_run('load-data', name='celeba', batch_size=bsnum, seed=11, size=64, **{'_dataset_mod': 'interpolated'})
else:
    assert False
len(dataset)


batch = dataset.get_batch()
X, *other = batch
X.shape

zs, ps = encode(X, distr=False, prior=True, eps_table=None)
def gen_eps_table(N, t=1., expand=True, gen=None):
    if not isinstance(N, int):
        N = N.shape[0]
    if expand:
        return [torch.randn(1, *z.shape[1:], generator=gen).mul(t).expand(N, *z.shape[1:]).to(z) for z in zs]
    return [torch.randn(N,*z.shape[1:], generator=gen).mul(t).to(z) for z in zs]
rec = decode(zs)
util.plot_imgs(rec);

# Xc = X[[torch.randint(len(X), size=()).item()]].expand(*X.shape)
# util.plot_imgs(decode(encode(Xc, distr=True, eps_table=None), eps_table=None));

# _global_eps_table = gen_eps_table(1, t=1, gen=torch.Generator().manual_seed(101))



q = enc(X)
len(q)

r = dec(q)
r.shape

util.plot_imgs(X);
util.plot_imgs(r);

plt.show()


print('done')

# Xc = X[:1].expand(8, *X.shape[1:])
# Xc.shape
#
# ds = encode(Xc, t=1., distr=True, prior=False)
#
# eps = [d.eps for d in ds]
#
# ds[0]
#
# # torch.use_deterministic_algorithms(True)
# # torch.manual_seed(101)
# # np.random.seed(101)
# # random.seed(101)
#
#
# # d2 = encode(Xc, t=0., distr=True, prior=False )
# d2 = encode(Xc, t=1., distr=True, prior=False, eps_table =eps )
# len(d2)
#
#
# # torch.manual_seed(101)
# # np.random.seed(101)
# # random.seed(101)
#
# # d3 = encode(Xc, t=0., distr=True, prior=False )
#
# d3 = encode(Xc, t=1., distr=True, prior=False, eps_table = eps)
# len(d3)
#
#
# r2 = decode(d2, eps_table=eps)
# dr2 = encode(r2, distr=True, eps_table=eps)
#
#
# r3 = decode(d3, eps_table=eps)
# dr3 = encode(r3, distr=True, eps_table=eps)
#
#
#
#
#
# print(len(X))



