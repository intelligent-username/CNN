"""
Evaluate SynthText OCR model on a test subset.
"""

import torch
from loader import build_loaders
from model import SynthText_CRNN
from text import tokenize_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build loaders (loader.py handles the collate/padding internally now)
    batch_size = 32
    _, _, test_loader, _ = build_loaders(
        batch_size=batch_size,
        num_workers=4,
        use_test=True,
        max_cap=12000  # Loads only first 12k images for indexing
    )

    # Load Model
    model = SynthText_CRNN(max_steps=21)
    # Fixed typo in filename: ATTn -> ATTN
    checkpoint_path = "../models/OCR_ATTN.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Please train first.")
        return

    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0
    total_chars = 0
    correct_chars = 0
    processed_images = 0
    max_test_images = 10000

    print(f"Model on GPU: {next(model.parameters()).is_cuda}")
    print("Starting evaluation...")

    with torch.no_grad():
        for crops, texts in test_loader:
            if crops is None or len(texts) == 0:
                continue

            # Limit evaluation to ~10k images
            if processed_images >= max_test_images:
                break
            
            # 1. Inputs: Loader now provides tensors (B, C, H, W), just move to GPU
            batch_inputs = crops.to(device)
            
            # 2. Targets: Tokenize manually, cast to Tensor, and stack
            tokenized_list = [torch.tensor(tokenize_text(t, max_len=model.max_steps), dtype=torch.long) for t in texts]
            batch_targets = torch.stack(tokenized_list).to(device) # (B, T)

            # 3. Forward Pass
            outputs = model(batch_inputs)  # (T, B, C)
            
            # 4. Reshape for Loss/Acc
            # Permute to (B, T, C) to match targets (B, T)
            outputs = outputs.permute(1, 0, 2) 
            B, T, C = outputs.shape
            
            # Calculate Loss
            loss = criterion(outputs.reshape(B*T, C), batch_targets.reshape(B*T))
            total_loss += loss.item() * B

            # Calculate Accuracy
            # preds: (B, T)
            preds = outputs.argmax(dim=2)
            
            # Mask out padding (index 0) so we don't count it as "correct" or "incorrect"
            mask = batch_targets != 0
            correct_chars += (preds[mask] == batch_targets[mask]).sum().item()
            total_chars += mask.sum().item()
            
            processed_images += B

    if total_chars > 0:
        avg_loss = total_loss / processed_images
        accuracy = correct_chars / total_chars
        print(f"Evaluated {processed_images} images.")
        print(f"Final Test Loss: {avg_loss:.4f}")
        print(f"Character Accuracy: {accuracy:.2%}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
