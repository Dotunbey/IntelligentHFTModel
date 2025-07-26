# Exports the HFT model to ONNX for MT5 integration
import torch
from src.models import IntelligentHFTModel

def export_to_onnx(config, model_path="gold_checkpoints/best.pth", output_path="hft_model.onnx"):
    # Load model
    model = IntelligentHFTModel(config).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    _, _ = model.load_best_checkpoint(model, optimizer)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, config['seq_len'], config['feature_dim']).to('cpu')
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['q_values', 'uncertainty', 'attentions'],
        dynamic_axes={'input': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}, 'uncertainty': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    export_to_onnx(config)
