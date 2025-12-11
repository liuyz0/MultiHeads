# first try
# CPU training
# 3 digits addition
# choose 4 edges to scan lr
# remove pretraning, which should not matter

# import necessary libraries
from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
import sys
import numpy as np
import time

task_id = int(sys.argv[1]) # from 0 to 39
num_tasks = int(sys.argv[2]) # should be 4 * 10 = 40
assert num_tasks == 40

head_hid_pairs = [(12, 384), (12, 24), (2, 24), (2, 384)]
num_head, hid_dim = head_hid_pairs[task_id // 10]
lr_range = torch.linspace(np.log(3e-5), np.log(3e-3), steps=10).exp()
lr = lr_range[task_id % 10]

num_steps = 0 # pretraining
num_epochs = 10 # fine tuning
eval_interval = 100

# define tokenizer and dataset
tokenizer = PreTrainedTokenizerFast.from_pretrained("./addition_tokenizer")
class DigitAdditionDatasetAllDigits(Dataset):
    def __init__(self, num_samples, k, tokenizer, pair_base: int = 0):
        self.num_samples = num_samples
        assert num_samples % k == 0, "num_samples must be multiple of k"
        self.k = k
        self.tokenizer = tokenizer
        self.pair_base = pair_base # evaluation offset to avoid overlap between train and eval
        
    def __len__(self):
        return self.num_samples # Each sample generates k examples

    def _get_digits(self, num: int):
        # convert numbers into a reversed list of digits, such that index 0 refers to the ones digit
        digits = str(num)
        if len(digits) < self.k:
            padding = "0" * self.k 
            digits = padding[:self.k-len(digits)] + digits
        return list(digits)
        
    def __getitem__(self, idx):
        # Determine which pair and which digit
        pair_idx = idx // (self.k) + self.pair_base
        digit_pos = idx % (self.k)
        
        # Use pair_idx as seed for reproducibility within epoch
        rng = np.random.RandomState(pair_idx)
        max_number = 10**self.k - 1
        a1_int = rng.randint(0, max_number)
        a2_int = rng.randint(0, max_number)
        A_int = a1_int + a2_int

        a1 = " ".join(self._get_digits(a1_int))
        a2 = " ".join(self._get_digits(a2_int))
        A = self._get_digits(A_int)[::-1]

        input_str = f"A = {a1} + {a2} , A {digit_pos} = ?"
        output = int(A[digit_pos])
        sent = f"Input: {input_str} Output: {output}"
        
        # Tokenize
        encoding = self.tokenizer(sent, return_tensors='pt', padding=False, add_special_tokens=False)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': output
        }
    
train_dataset = DigitAdditionDatasetAllDigits(num_samples=32*6000, k=3, tokenizer=tokenizer) # 6000 steps
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
eval_dataset = DigitAdditionDatasetAllDigits(num_samples=64*9, k=3, tokenizer=tokenizer, pair_base=32*2000)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)

# define model
# helper
ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
}


@dataclass
class AttentionConfig:
    D: int = 768
    layer_idx: Optional[int] = None
    n_heads: int = 4
    causal: bool = True
    device: str = "cuda"


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class Attention(nn.Module):  # BSD -> BSD
    def __init__(self, layer_idx: int, config: AttentionConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.D = config.D
        self.n_heads = config.n_heads
        assert self.D % self.n_heads == 0
        self.head_dim = self.D // self.n_heads
        self.Wq = nn.Linear(self.D, self.D, bias=False)
        self.Wk = nn.Linear(self.D, self.D, bias=False)
        self.Wv = nn.Linear(self.D, self.D, bias=False)
        self.causal = config.causal
        #self.Wo = nn.Linear(self.D, self.D, bias=False)
        #self.Wo.weight.data.zero_()  # initialize to zero for stability
        self.W_O = nn.Parameter(torch.zeros(self.n_heads, self.head_dim, self.D))
        self.device = config.device
        # Hook points
        self.hook_attn_pattern = HookPoint()
        self.hook_attn_output_per_head = HookPoint()

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None
    ) -> torch.Tensor:  # input is [B, S, D]
        B, S, D = x.shape

        # Make each QKV [B, S, D] --> [B, nh, S, hd]
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)  # all [B, S, D]

        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, S, hd]
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # update kv cache
        layer_idx = self.layer_idx
        if kv_cache is not None and layer_idx is not None:
            # its preallocated, just write to the memory of the cache using state of current_length
            kv_cache.update(layer_idx, K, V)
            K = kv_cache.keys[layer_idx][:, :, : kv_cache.current_length, :]
            V = kv_cache.values[layer_idx][:, :, : kv_cache.current_length, :]

        # [B, nh, S, hd] @ [B, nh, hd, S] -> [B, nh, S, S]
        scale = torch.sqrt(
            torch.tensor(self.head_dim, dtype=Q.dtype, device=self.device)
        )
        logits = (Q @ K.transpose(-2, -1)) / scale
        if self.causal:
            mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
            logits_masked = logits.masked_fill(mask, float("-inf"))
        else:
            logits_masked = logits

        A = F.softmax(logits_masked, dim=-1)  # [B, nh, S, S]
        # Hook attention pattern: [B, nh, S, S]
        A = self.hook_attn_pattern(A)

        preout = torch.einsum(
            "bnxy,bnyd->bnxd", A, V
        )  # [B, nh, S, hd]

        # Rearrange W_O from [D, D] to [nh, hd, D]
        #W_O = self.Wo.weight.T.view(self.n_heads, self.head_dim, self.D)
        attn_output_per_head = torch.einsum(
            "bnxd,ndh->bnxh", preout, self.W_O
        )  # [B, nh, S, D]
        # Reorder to [B, S, nh, D] and hook
        attn_output_per_head_seq = attn_output_per_head.transpose(1, 2)
        attn_output_per_head_seq = self.hook_attn_output_per_head(attn_output_per_head_seq)
        # Sum across heads -> [B, S, D]
        attn_out = attn_output_per_head_seq.sum(dim=2)
        return attn_out  # [B, S, D]


@dataclass
class MLPConfig:
    D: int
    hidden_multiplier: int = 4
    act: str = "gelu"
    device: Optional[torch.device] = None


# most important fact about MLP: it operates on each token independently, ie. D --> D
class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.D = config.D
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.up_proj = nn.Linear(self.D, self.D * config.hidden_multiplier, bias=False)
        self.down_proj = nn.Linear(
            self.D * config.hidden_multiplier, self.D, bias=False
        )
        self.down_proj.weight.data.zero_()  # initialize to zero for stability
        self.act = ACT2FN[config.act]
        # Hook point at MLP mid activation
        self.hook_mlp_mid = HookPoint()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # BSD -> BSD automatically on last dim
        mid = self.act(self.up_proj(x))
        mid = self.hook_mlp_mid(mid)  # [B, S, D*mult]
        return self.down_proj(mid)


@dataclass
class LNConfig:
    D: int
    eps: float = 1e-9
    device: Optional[torch.device] = None


class LN(nn.Module):
    def __init__(self, config: LNConfig):
        super().__init__()
        self.D = config.D
        self.eps = config.eps
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mean_scale = nn.Parameter(torch.zeros(self.D))
        self.std_scale = nn.Parameter(torch.ones(self.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        mean = x.mean(dim=-1, keepdim=True)  # [B, S, 1]
        std = (x.var(dim=-1, keepdim=True) + self.eps) ** 0.5  # [B, S, 1]
        x_norm = (x - mean) / (std)
        return x_norm * self.std_scale + self.mean_scale


@dataclass
class TransformerLayerConfig:
    D: int = 768
    n_heads: int = 4
    device: Optional[torch.device] = None


class TransformerLayer(nn.Module):
    def __init__(self, layer_idx: int, config: TransformerLayerConfig):
        super().__init__()
        self.D = config.D
        self.layer_idx = layer_idx
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        attn_config = AttentionConfig(
            D=self.D, n_heads=config.n_heads, device=self.device
        )
        mlp_config = MLPConfig(D=self.D, device=self.device)
        ln_config = LNConfig(D=self.D, device=self.device)

        self.attn = Attention(self.layer_idx, attn_config)
        self.mlp = MLP(mlp_config)
        self.ln1 = LN(ln_config)
        self.ln2 = LN(ln_config)
        # Residual stream hook points
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None, return_attn: bool = False
    ) -> torch.Tensor:  # x is BSD
        x = self.hook_resid_pre(x)
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, kv_cache=kv_cache)
        x = x + attn_out
        x = self.hook_resid_mid(x)
        ln2_out = self.ln2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out
        x = self.hook_resid_post(x)
        if return_attn:
            return x, attn_out
        return x


@dataclass
class PositionalEmbeddingConfig:
    max_seq_len: int
    D: int
    device: Optional[torch.device] = None


class PositionalEmbedding(nn.Module):
    def __init__(self, config: PositionalEmbeddingConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        B, S, D = x.shape
        return x + self.pos_embedding[:S]  # Broadcasting handles batch dimension


@dataclass
class EmbeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None


class EmbeddingLayer(nn.Module):
    def __init__(self, config: EmbeddingLayerConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.embedding = nn.Parameter(torch.randn(config.vocab_size, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


@dataclass
class UnembeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None


class UnembeddingLayer(nn.Module):
    def __init__(self, config: UnembeddingLayerConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.V = config.vocab_size
        self.unembedding = nn.Linear(config.D, self.V, bias=False)
        self.unembedding.weight.data.zero_() # initialize to zero for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        return self.unembedding(x)


@dataclass
class TransformerConfig:
    hidden_dim: int = 768
    depth: int = 2
    n_heads: int = 4
    vocab_size: int = 50257
    max_seq_len: int = 128
    device: Optional[torch.device] = None


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.depth = config.depth
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        emb_config = EmbeddingLayerConfig(
            vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device
        )
        pos_emb_config = PositionalEmbeddingConfig(
            max_seq_len=config.max_seq_len, D=config.hidden_dim, device=config.device
        )
        unemb_config = UnembeddingLayerConfig(
            vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device
        )

        self.emb = EmbeddingLayer(emb_config)
        self.pos_emb = PositionalEmbedding(pos_emb_config)

        self.ln_final = LN(LNConfig(D=config.hidden_dim, device=config.device))
        self.unemb = UnembeddingLayer(unemb_config)

        layer_config = TransformerLayerConfig(
            D=config.hidden_dim, n_heads=config.n_heads, device=config.device
        )
        self.layers = nn.ModuleList(
            [TransformerLayer(idx, layer_config) for idx in range(config.depth)]
        )
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i

        self.device = config.device

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None, return_attn: bool = False
    ) -> torch.Tensor:
        x = self.emb(x)
        if kv_cache is not None:
            # When decoding, only add positional embeddings for the new tokens.
            pos_offset = kv_cache.current_length
            pos_emb = self.pos_emb.pos_embedding[
                pos_offset : pos_offset + x.size(1)
            ].unsqueeze(0)
            x = x + pos_emb
        else:
            x = self.pos_emb(x)

        all_attn = []
        for _, layer in enumerate(self.layers):
            if return_attn:
                x, attn = layer(x, kv_cache=kv_cache, return_attn=True)
                all_attn.append(attn)
            else:
                x = layer(x, kv_cache=kv_cache)

        x = self.ln_final(x)
        logits = self.unemb(x)
        if return_attn:
            return logits, torch.stack(all_attn, dim=0)
        return logits
    
cfg = TransformerConfig(
    hidden_dim=hid_dim,
    depth=2,
    n_heads=num_head,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=19,
    device="cpu",
)
model = Transformer(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# training loop
pre_train_losses = []
pre_eval_losses = []
# later version save the model checkpoints
start_time = time.time()

# pretraining
for step, batch in enumerate(train_dataloader):
    model.train()
    input_ids = batch['input_ids']
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), 
                           input_ids[:, 1:].contiguous().view(-1))
    pre_train_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % eval_interval == 0:
        model.eval()
        total_eval_loss = 0.0
        for eval_batch in eval_dataloader:
            with torch.no_grad():
                eval_input_ids = eval_batch['input_ids']
                eval_logits = model(eval_input_ids)
                total_eval_loss += F.cross_entropy(eval_logits[:, -2, :], eval_input_ids[:, -1]).item()

        pre_eval_losses.append(total_eval_loss / len(eval_dataloader))
        print(f"Step {step+1}/{num_steps}, Train Loss: {loss.item():.4f}, Eval Loss: {pre_eval_losses[-1]:.4f}, Time Elapsed: {time.time() - start_time:.2f}s")

    if step + 1 >= num_steps:
        break

# fine tuning focus on last digit prediction
train_losses = []
eval_losses = []
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # reset optimizer
step = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()
        input_ids = batch['input_ids']
        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, -2, :], input_ids[:, -1])
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            model.eval()
            total_eval_loss = 0.0
            for eval_batch in eval_dataloader:
                with torch.no_grad():
                    eval_input_ids = eval_batch['input_ids']
                    eval_logits = model(eval_input_ids)
                    total_eval_loss += F.cross_entropy(eval_logits[:, -2, :], eval_input_ids[:, -1]).item()    

            eval_losses.append(total_eval_loss / len(eval_dataloader))
            print(f"Epoch {epoch+1}, Step {step+1}, Train Loss: {loss.item():.4f}, Eval Loss: {eval_losses[-1]:.4f}, Time Elapsed: {time.time() - start_time:.2f}s")
        
        step += 1

# save the results
torch.save({
    'pre_train_losses': pre_train_losses,
    'pre_eval_losses': pre_eval_losses,
    'train_losses': train_losses,
    'eval_losses': eval_losses}, f'../../../orcd/pool/MultiHeads/outputs/exp-1-1-{task_id}.pt')