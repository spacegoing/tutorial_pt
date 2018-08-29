# -*- coding: utf-8 -*-
import time
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# (batchsize, T, feat_dim)
N = 20000
T = 4
feat_dim = 40
out_dim = 1
x = torch.arange(N * T * feat_dim, device=device).view(N, T, feat_dim)
y = x.matmul(torch.ones(feat_dim, out_dim, device=device))

model = nn.Linear(feat_dim, out_dim)
model.to(device)
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters())


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train_epoch(x, y, model, opt, criterion):
  '''
  view version:
    model1 = nn.Linear(4, 1)
    opt1 = optim.Adam(model1.parameters())
    wrong: zero_grad should start from each epoch
    model1.zero_grad()
    for i in range(count):
      res1 = model1(x.view(-1,4))
      loss1 = criterion(res1, y.view(-1).unsqueeze(1))
      loss1.backward()
      opt1.step()
    np.save('t_1',model1.weight.detach().numpy())

    t = np.load('t.npy')
    t1 = np.load('t_1.npy')
    t == t1
  '''
  epoch_no = 200000
  epoch_losses = []
  batchsize = 200

  start = time.time()
  print_every = 1000

  for i in range(1, epoch_no + 1):
    model.zero_grad()
    res = model(x)
    loss = criterion(res, y)
    loss.backward()
    opt.step()

    epoch_losses.append(loss.item())

    if i % print_every == 0:
      print('%s (%d %d%%) %.4f' % (timeSince(start, i / epoch_no), i,
                                   i / epoch_no * 100, epoch_losses[i - 1]))

  return epoch_losses


res = train_epoch(x, y, model, opt, criterion)
'''
res[3500]
Out[120]: 0.009238757193088531
Out[150]: 0.009235935285687447
'''
