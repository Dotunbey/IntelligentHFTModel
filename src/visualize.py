# Visualizes attention weights and key ticks influencing cx7
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import IntelligentHFTModel
from src.features import IntelligentFeatureExtractor
from src.utils import fetch_tick_data

def visualize_attention(config, tick_data, model_path="gold_checkpoints/best.pth"):
    # Load model and data
    model = IntelligentHFTModel(config).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    _, _ = model.load_best_checkpoint(model, optimizer)
    feature_extractor = IntelligentFeatureExtractor()
    
    # Process one tick for visualization
    feature_extractor.update_buffers({
        'price': tick_data['price'][config['seq_len']],
        'volume': tick_data['volume'][config['seq_len']],
        'order_flow': tick_data['order_flow'][config['seq_len']],
        'bid': tick_data['bid'][config['seq_len']],
        'ask': tick_data['ask'][config['seq_len']],
        'timestamp': tick_data['timestamp'][config['seq_len']]
    })
    features = feature_extractor.extract_intelligent_features()
    if features is None:
        print("Insufficient data for visualization")
        return
    
    # Predict with attention
    market_data = {
        'features': torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cpu'),
        'prices': torch.tensor(list(feature_extractor.price_buffer), dtype=torch.float32).unsqueeze(0).to('cpu'),
        'volumes': torch.tensor(list(feature_extractor.volume_buffer), dtype=torch.float32).unsqueeze(0).to('cpu'),
        'order_flow': torch.tensor(list(feature_extractor.order_flow_buffer), dtype=torch.float32).unsqueeze(0).to('cpu')
    }
    with torch.no_grad():
        _, _, attentions = model(market_data['features'])
    
    # Visualize attention weights
    for layer, attn in enumerate(attentions):
        attn = attn[0].cpu().numpy()  # First batch, CPU
        plt.figure(figsize=(10, 8))
        plt.imshow(attn.mean(axis=1), cmap='viridis')
        plt.title(f'Attention Weights - Layer {layer}')
        plt.colorbar()
        plt.savefig(f"gold_checkpoints/attention_layer_{layer}.png")
        plt.close()
    
    # Identify key ticks influencing cx7
    cx7 = features[-5]
    key_ticks = np.argsort(np.abs(features))[-4:][::-1]
    print(f"Key ticks influencing cx7 ({cx7:.4f}): {key_ticks.tolist()}")

if __name__ == "__main__":
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    tick_data = fetch_tick_data(use_sample=True)
    visualize(config, tick_data)
