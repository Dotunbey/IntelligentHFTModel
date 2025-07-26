# Defines the Transformer-based HFT model with DQN, optimized for CPU
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)  # Added to prevent overfitting on CPU

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.query(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc_out(out)
        return out, attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended, attention = self.attention(x, x, x, mask)
        x = self.norm1(attended + x)
        ff = self.feed_forward(x)
        x = self.norm2(ff + x)
        return x, attention

class IntelligentHFTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_fc = nn.Linear(config['feature_dim'], config['hidden_dim'])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config['hidden_dim'], config['n_heads'], dropout=0.2)
            for _ in range(config['n_layers'])
        ])
        self.fc = nn.Linear(config['hidden_dim'], config['action_dim'])
        self.uncertainty_head = nn.Linear(config['hidden_dim'], config['action_dim'])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        x = self.input_fc(x)
        x = self.dropout(torch.relu(x))
        attentions = []
        for block in self.transformer_blocks:
            x, attention = block(x, mask)
            attentions.append(attention)
        q_values = self.fc(x[:, -1, :])
        uncertainty = torch.softmax(self.uncertainty_head(x[:, -1, :]), dim=-1)
        return q_values, uncertainty, attentions

    def predict_with_confidence(self, market_data):
        # Run inference on CPU
        features = market_data['features'].to('cpu')
        q_values, uncertainty, _ = self.forward(features)
        action = torch.argmax(q_values, dim=-1).item()
        confidence = torch.max(uncertainty, dim=-1)[0].item()
        return {'action': action, 'confidence': confidence}
