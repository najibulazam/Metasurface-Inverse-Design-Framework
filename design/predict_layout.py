# design/predict_layout.py
import os
import numpy as np
import pandas as pd
import torch
import json
from model.vit_small import VisionTransformerSmall

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_inverse_checkpoint(device):
    candidate_paths = [
        os.path.join(CKPT_DIR, "inverse_model_final.pt"),
        os.path.join(CKPT_DIR, "inverse_fine_tune_final.pt"),
    ]
    for ckpt_path in candidate_paths:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]
            return ckpt_path, checkpoint
    raise FileNotFoundError(
        "Fine-tuned model checkpoint not found. Checked: "
        + ", ".join(candidate_paths)
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for One-Shot Prediction...")

    # ১. টার্গেট জোন্স ম্যাট্রিক্স লোড করা
    jm_path = os.path.join(DATASET_DIR, "metalens_target_JM.npy")
    if not os.path.exists(jm_path):
        raise FileNotFoundError(f"Target file not found at {jm_path}. Run design.metalens_design first.")
    
    target_jm = np.load(jm_path)  # Shape: [64, 64, 11, 6]
    size_x, size_y, n_wave, n_ch = target_jm.shape
    
    # ২. মডেল লোড করা (num_para=6 দিয়ে ইনভার্স হেড একটিভ করা)
    model = VisionTransformerSmall(img_size=(n_wave, n_ch), num_para=6)
    ckpt_path, state_dict = load_inverse_checkpoint(device)
    model.load_state_dict(state_dict)
    print(f"Loaded fine-tuned Inverse Design AI Model successfully from: {ckpt_path}")
        
    model.to(device)
    model.eval()

    # ৩. ডাটাকে ফ্ল্যাট করে ব্যাচ আকারে সাজানো (ViT প্রসেস করার জন্য)
    # [64, 64, 11, 6] -> [4096, 1, 11, 6]
    flat_jm = target_jm.reshape(-1, n_wave, n_ch)
    input_tensor = torch.from_numpy(flat_jm).unsqueeze(1).float().to(device)

    # ৪. ওয়ান-শট প্রেডিকশন
    print(f"Predicting structural configurations for all {len(input_tensor)} pixels...")
    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy() # Shape: [4096, 6]

    # ৫. ডাটা ডিনরমালাইজেশন (এআই ম্যাট্রিক্স থেকে ফিজিক্যাল ন্যানোমিটারে রূপান্তর)
    # যেহেতু আমরা জোন্স ম্যাট্রিক্স থেকে ডাইরেক্ট প্রেডিক্ট করছি, তাই ডাটার বাউন্ডারি ঠিক রাখা
    # x, y সাইজ ৮০nm থেকে ১৬০nm এবং কোণ ০ থেকে ৮০ ডিগ্রির মধ্যে রেসট্রিক্ট করা হলো
    predictions[:, [0, 1, 3, 4]] = np.clip(predictions[:, [0, 1, 3, 4]], 80, 160)
    predictions[:, [2, 5]] = np.clip(predictions[:, [2, 5]], 0, 80)

    # ৬. ফলাফল CSV আকারে সেভ করা
    columns = ["x1_nm", "y1_nm", "angle1_deg", "x2_nm", "y2_nm", "angle2_deg"]
    df = pd.DataFrame(predictions, columns=columns)
    
    # পিক্সেল পজিশন অ্যাড করা (Grid X, Grid Y)
    grid_y, grid_x = np.mgrid[0:size_y, 0:size_x]
    df.insert(0, "grid_x", grid_x.flatten())
    df.insert(1, "grid_y", grid_y.flatten())

    output_csv = os.path.join(DATASET_DIR, "predicted_metalens_layout.csv")
    df.to_csv(output_csv, index=False)
    print(f"🎉 Success! Predicted physical structures saved to: {output_csv}")

if __name__ == "__main__":
    main()
