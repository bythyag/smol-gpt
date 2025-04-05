import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#------------------------------------
# GPT-2 Model Definition
#------------------------------------

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        normalized_input = (input - mean) / torch.sqrt(var + 1e-5)
        output = normalized_input * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                      .view(1, 1, config['block_size'], config['block_size']))
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.dropout = config['dropout']

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu    = nn.GELU() # Changed from ReLU to GELU (standard in GPT-2)
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['vocab_size'] != -1, "Vocab size must be set in config"
        assert config['block_size'] > 0, "Block size must be set in config"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config['n_layer']))

        # Report number of parameters
        print(f"Model Parameter Count: {self.get_num_params()/1e6:.2f} M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (weights tied), we count params of wte and exclude lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract lm_head params if weights are tied
            n_params -= self.transformer.wpe.weight.numel() # wpe is always separate
            # The tied weights are in transformer.wte.weight
            # n_params -= self.lm_head.weight.numel() <--- This was double counting wte
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        if t > self.config['block_size']:
             # Crop idx if sequence length is greater than block size
             idx = idx[:, -self.config['block_size']:]
             t = self.config['block_size']
             # If targets are provided, they also need cropping
             if targets is not None:
                 targets = targets[:, -self.config['block_size']:]

        #assert t <= self.config['block_size'], f"Cannot forward seq len {t}, block size {self.config['block_size']}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(idx) # shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape (b, t, n_embd)

        if targets is not None:
            logits = self.lm_head(x) # shape (b, t, vocab_size)
            # Ensure targets are long type for cross_entropy
            targets = targets.long()
            # Handle potential PAD tokens if using ignore_index (using 0 for word-level pad)
            ignore_idx = 0 if self.config['level'] == 'word' else -1 # Don't ignore anything for char
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_idx)
        else:
            # Inference: only compute logits for the last time step
            logits = self.lm_head(x[:, [-1], :]) # shape (b, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval() # Set model to evaluation mode
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond) # Forward pass
            logits = logits[:, -1, :] / temperature # Get last timestep, apply temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Apply top-k

            probs = F.softmax(logits, dim=-1) # Get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # Sample next token
            idx = torch.cat((idx, idx_next), dim=1) # Append to sequence
        self.train() # Set model back to train mode
        return idx