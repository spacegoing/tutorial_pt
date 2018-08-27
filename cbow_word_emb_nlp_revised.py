# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMB_SIZE = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
  context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
  target = raw_text[i]
  data.append((context, target))
  print(data[:5])


def make_context_vector(context, word_to_ix):
  idxs = [word_to_ix[w] for w in context]
  return torch.tensor([idxs], dtype=torch.long)


# create your model and train.  here are some functions to help you make
# the data ready for use by your module


class CBOW(nn.Module):

  def __init__(self, vocab_size, embed_size):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    self.linear = nn.Linear(embed_size, vocab_size)

  def forward(self, inputs):
    '''
    inputs: 3D (N,Context_Size,Embed_Size)
      because later functions like nllloss() must input 2D mat,
      (N,C) (samples,classes), so keep dim
    '''
    emb_mat = self.embeddings(inputs)
    emb_vec = emb_mat.sum(dim=0)
    log_probs = self.linear(emb_vec)
    return F.log_softmax(log_probs)


model = CBOW(len(word_to_ix), EMB_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# loss
losses = []  # record losses
loss_fun = nn.NLLLoss()

for epoch in range(10):
  total_loss = 0
  for sent, target in data:
    sent_ids = make_context_vector(sent, word_to_ix)
    model.zero_grad()

    pred = model(sent_ids)
    # pred shape: (N,C)
    loss = loss_fun(pred, torch.tensor([word_to_ix[target]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  losses.append(total_loss)
