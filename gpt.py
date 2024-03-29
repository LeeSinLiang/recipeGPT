import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from utils import top_k_top_p_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiInputSequential(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs

class HeadAttention(nn.Module):
	def __init__(self, head_size, embedding_dim, max_length, pDropout):
		super().__init__()
		self.query = nn.Linear(embedding_dim, head_size, bias=False)
		self.key = nn.Linear(embedding_dim, head_size, bias=False)
		self.value = nn.Linear(embedding_dim, head_size, bias=False)
		# not a model parameter, just a variable, hence called as buffer. Have to call register_buffer to denote it
		self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length))) # Decoder block: prevent future tokens from talking to current tokens. Only past tokens can talk to current tokens.
		self.head_size = head_size
		self.dropout = nn.Dropout(pDropout)

	def forward(self, x, attention_mask=None):
		B, T, C = x.shape
		qK = self.query(x) @ self.key(x).transpose(-2, -1) * (1.0 / math.sqrt(self.head_size)) # change to self.head_size # scaled attention (normilization) by dividing it
		if attention_mask is not None: # doesnt work, rows of -inf -> nan in cross entropy loss
			qK = attention_mask.unsqueeze(2) * qK
			qK = qK.masked_fill(qK == 0, float('-inf'))
		qK = qK.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		qK = F.softmax(qK, dim = -1)
		qK = self.dropout(qK)
		qK = qK.masked_fill(torch.isnan(qK), 0) # testing
		out = qK @ self.value(x)
		return out

class MultiHeadAttention(nn.Module):
	# mulitple heads of self attention in parallel

	def __init__(self, num_heads, head_size, n_emb, max_length, pDropout):
		super().__init__()
		# self.heads = nn.ModuleList([HeadAttention(head_size, embedding_dim, max_length) for _ in range(num_heads)])
		self.heads = nn.ModuleList([HeadAttention(head_size, n_emb, max_length, pDropout) for _ in range(num_heads)])
		self.projection = nn.Linear(n_emb, n_emb) # linear transformation of self attention
		self.dropout = nn.Dropout(pDropout)

	def forward(self, x, attention_mask):
		out = torch.cat([h(x, attention_mask) for h in self.heads], dim=-1)
		out = self.dropout(self.projection(out))
		return out

class FeedForward(nn.Module):
	def __init__(self, n_emb, pDropout=0.2):
		super().__init__()
		self.out = nn.Sequential(
			nn.Linear(n_emb, 4 * n_emb),
			nn.ReLU(),
			nn.Linear(4 * n_emb, n_emb),
			nn.Dropout(pDropout)
		)

	def forward(self, x):
		return self.out(x)

class Block(nn.Module):
	def __init__(self, n_head, n_emb, max_length, pDropout):
		super().__init__()
		head_size = n_emb // n_head
		self.sa_head = MultiHeadAttention(n_head, head_size, n_emb, max_length, pDropout) # i.e. 4 heads of 8 dimensional self-attention, which concatenates to 32 (embedding_dim)
		self.feed_fwd = FeedForward(n_emb)
		self.ln1 = nn.LayerNorm(n_emb) # pre-normalization
		self.ln2 = nn.LayerNorm(n_emb)

	def forward(self, x, attention_mask):
		# x + self... is to fork the computation outside and join back (skip connection)
		x = x + self.sa_head(self.ln1(x), attention_mask) # token communication each other
		x = x + self.feed_fwd(self.ln2(x)) # token indiviually think
		return x, attention_mask

class GPT(nn.Module):
	def __init__(self, vocab_size, max_length, n_emb, n_head, n_layer, pDropout):
		super().__init__()
		self.embedding_table = nn.Embedding(vocab_size, n_emb)
		self.positional_embedding = nn.Embedding(max_length, n_emb) # to know my tokens location
		self.blocks = MultiInputSequential(*[Block(n_head, n_emb, max_length, pDropout) for _ in range(n_layer)])
		self.ln_final = nn.LayerNorm(n_emb)
		self.lm_head = nn.Linear(n_emb, vocab_size) # original is embedding_dim
		self.max_length = max_length
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, attention_mask=None, target=None):
		B, T = idx.shape

		tok_emb = self.embedding_table(idx) # (B, T, C)
		pos_emb = self.positional_embedding(torch.arange(T, device=device)) # torch.arange(T) -> [0, 1,... T] -> (T, C)
		x = tok_emb + pos_emb # (B, T, C)
		x, _ = self.blocks(x, attention_mask)
		logits = self.lm_head(x) # (B, T, vocab_size)

		if target == None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T,C)
			target = target.view(-1)  # (B*T)
			if attention_mask is not None:
				loss = F.cross_entropy(logits, target, ignore_index=1) # ignore pad_token
			else:
				loss = F.cross_entropy(logits, target)
		return (logits, loss)

	def generate(self, tokenizer, out_fn, progress, idx: Tensor, max_tokens_generate: int, temperature: int = 1, top_k: int = 0.0, top_p: float = 1.0):
		# idx (B, T)
		for count in range(max_tokens_generate):
			# crop it to get latest <max_length> tokens since pos_emb only has max_length size
			idx_condition = idx[:, -self.max_length:]
			logits, loss = self.forward(idx_condition)
			# get the last character logits to predict the next character, (B, C)
			logits = logits[:, -1, :] / temperature
			logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

			probs = F.softmax(logits, dim=1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx_item = idx_next.item()
			if (idx_item == 0):
				progress_bar, progress_text = progress
				progress_bar.progress(100, text=progress_text)
				break
			out_fn(tokenizer.decode(idx_item))
			idx = torch.cat((idx, idx_next), dim=1)
			progress_bar, progress_text = progress
			progress_bar.progress(math.floor(
				((count+1)/max_tokens_generate)*100), text=progress_text)
		progress_bar, _ = progress
		progress_bar.progress(100, text='Operation Successful!')
		return idx.view(idx.shape[1], )
