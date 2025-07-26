# Trains the HFT model using DQN, optimized for CPU with smaller replay buffer
import torch
import torch.optim as optim
from src.models import IntelligentHFTModel
from src.features import IntelligentFeatureExtractor
from src.utils import fetch_tick_data, save_checkpoint, load_best_checkpoint
from collections import deque
import numpy as np

class DQN:
    def __init__(self, config):
        # Initialize model and target network on CPU
        self.model = IntelligentHFTModel(config).to('cpu')
        self.target_model = IntelligentHFTModel(config).to('cpu')
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.memory = deque(maxlen=10000)  # Smaller replay buffer for CPU
        self.gamma = 0.99
        self.batch_size = 16  # Reduced for CPU
        self.config = config

    def update(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        # Convert to CPU tensors
        states = torch.tensor(states, dtype=torch.float32).to('cpu')
        actions = torch.tensor(actions, dtype=torch.long).to('cpu')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cpu')
        next_states = torch.tensor(next_states, dtype=torch.float32).to('cpu')
        dones = torch.tensor(dones, dtype=torch.float32).to('cpu')
        
        # Compute loss
        q_values, _, _ = self.model(states)
        next_q_values, _, _ = self.target_model(next_states)
        target_q = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=-1)[0]
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(q_value, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        # Sync target network with model
        self.target_model.load_state_dict(self.model.state_dict())

def train(config, tick_data, epochs=30):
    # Train for fewer epochs on CPU
    model = DQN(config)
    feature_extractor = IntelligentFeatureExtractor()
    capital = 10000
    returns = []
    best_sharpe = -float('inf')
    
    for epoch in range(epochs):
        position = 0
        episode_return = 0
        for i in range(config['seq_len'], len(tick_data['price']) - 1):
            # Extract features
            feature_extractor.update_buffers({
                'price': tick_data['price'][i],
                'volume': tick_data['volume'][i],
                'order_flow': tick_data['order_flow'][i],
                'bid': tick_data['bid'][i],
                'ask': tick_data['ask'][i],
                'timestamp': tick_data['timestamp'][i]
            })
            features = feature_extractor.extract_intelligent_features()
            if features is None:
                continue
            state = np.array([features], dtype=np.float32)
            
            # Predict action
            pred = model.model.predict_with_confidence({'features': torch.tensor(state).unsqueeze(0)})
            action = pred['action']
            price_change = tick_data['price'][i+1] - tick_data['price'][i]
            reward = 0
            # Simulate trading
            if action == 1 and position != 1:
                position = 1
                reward = -(tick_data['ask'][i] - tick_data['bid'][i]) / capital
            elif action == 2 and position != -1:
                position = -1
                reward = -(tick_data['ask'][i] - tick_data['bid'][i]) / capital
            elif action == 0 and position != 0:
                reward = price_change * position / capital
                position = 0
            episode_return += reward
            
            # Update model
            next_features = feature_extractor.extract_intelligent_features()
            next_state = np.array([next_features], dtype=np.float32) if next_features is not None else state
            done = i == len(tick_data['price']) - 2
            loss = model.update(state, action, reward, next_state, done)
            
            if i % 1000 == 0:
                model.update_target()
        
        # Compute metrics
        returns.append(episode_return)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        model.scheduler.step(sharpe_ratio)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Sharpe: {sharpe_ratio:.2f}, LR: {model.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        metrics = {'profit': episode_return * capital, 'sharpe_ratio': sharpe_ratio, 'win_rate': np.mean(np.array(returns) > 0), 'max_drawdown': -400.0}
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            save_checkpoint(model.model, model.optimizer, epoch, metrics)

if __name__ == "__main__":
    # Config optimized for CPU
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    tick_data = fetch_tick_data(use_sample=True)
    train(config, tick_data)
