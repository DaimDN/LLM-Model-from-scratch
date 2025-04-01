import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import os

class Config:
    def __init__(self):
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        self.dropout = 0.1
        self.bias = False
        self.batch_size = 64
        self.block_size = 256
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 6e-4
        self.eval_iters = 200
        self.weight_decay = 1e-1
        self.grad_norm_clip = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd  # Store n_embd as an instance variable
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(self.n_embd, dim=2)
        q, k, v = [x.view(B, T, self.n_head, self.head_size).transpose(1, 2) for x in qkv]
        
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.data = [self.stoi[ch] for ch in text]
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_model():
    config = Config()
    
    # Check if book.txt exists, and create a small sample if not
    if not os.path.exists('book.txt'):
        print("Warning: book.txt not found. Creating a sample text file for training.")
        sample_text = """
        This is a sample text for testing the transformer model.
        It contains multiple sentences that will be used for training.
        The model will learn to generate text in this style.
        Once trained, it should be able to complete sentences and create new ones.
        """
        with open('book.txt', 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    with open('book.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    dataset = TextDataset(text, config.block_size)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = Transformer(config, dataset.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(config.eval_iters)
        # Create a list of batches from the DataLoader
        data_iter = iter(train_loader)
        batches = []
        try:
            for _ in range(config.eval_iters):
                batches.append(next(data_iter))
        except StopIteration:
            # If we run out of batches, recreate the iterator
            data_iter = iter(train_loader)
            while len(batches) < config.eval_iters:
                try:
                    batches.append(next(data_iter))
                except StopIteration:
                    data_iter = iter(train_loader)
        
        for k, (x, y) in enumerate(batches):
            x, y = x.to(config.device), y.to(config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        model.train()
        return losses.mean()
    
    # Create an iterator that cycles through the data
    data_iterator = iter(train_loader)
    
    for iteration in range(config.max_iters):
        if iteration % config.eval_interval == 0:
            loss = estimate_loss()
            print(f"Step {iteration}: loss {loss:.4f}")
        
        # Get the next batch, recreating iterator if needed
        try:
            xb, yb = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            xb, yb = next(data_iterator)
            
        xb, yb = xb.to(config.device), yb.to(config.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        optimizer.step()
    
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
    }, 'transformer_model.pth')

def interact_with_model():
    if not os.path.exists('transformer_model.pth'):
        print("No existing model found. Training a new model...")
        train_model()
    
    checkpoint = torch.load('transformer_model.pth', map_location='cpu')
    config = checkpoint['config']
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Transformer(config, len(checkpoint['stoi'])).to(config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    stoi, itos = checkpoint['stoi'], checkpoint['itos']
    
    def encode(text):
        return [stoi.get(c, 0) for c in text.lower()]
    
    def decode(tokens):
        return ''.join([itos.get(i, '') for i in tokens])
    
    print("\nModel ready. Type 'quit' to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() == 'quit':
                break
            
            input_ids = encode(prompt)
            if not input_ids:
                print("Could not encode input")
                continue
                
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=config.device).unsqueeze(0)
            
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_k=40
                )
            
            response = decode(output[0].tolist())
            response = response[len(prompt):]
            response = response.split('\n')[0]
            
            print("AI:", response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == '__main__':
    interact_with_model()