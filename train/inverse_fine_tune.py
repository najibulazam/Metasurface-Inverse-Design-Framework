import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.vit_small import VisionTransformerSmall

class FineTuneDataset(Dataset):
    def __init__(self, dataset_dir):
        jm_df = pd.read_csv(os.path.join(dataset_dir, "JM_dimer.csv"))
        param_df = pd.read_csv(os.path.join(dataset_dir, "param_dimer.csv"))
        self.jm_data = torch.from_numpy(jm_df.to_numpy()).view(-1, 1, 21, 6).float()
        self.params = torch.from_numpy(param_df.to_numpy()).float()

    def __len__(self):
        return self.jm_data.shape[0]

    def __getitem__(self, idx):
        return self.jm_data[idx], self.params[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")
    
    dataset = FineTuneDataset(dataset_dir)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # প্রি-ট্রেইনড মডেল লোড এবং ফাইন-টিউনিং হেড যোগ (৬টি প্যারামিটার প্রেডিকশনের জন্য)
    encoder = VisionTransformerSmall(img_size=(21, 6), patch_size=1, in_chans=1, embed_dim=512, depth=8, num_heads=16, num_para=6)
    
    # প্রি-ট্রেইনড ওজন লোড করা
    pretrained_dict = torch.load(os.path.join(ckpt_dir, "simmim_epoch50.pt"))["model"]
    model_dict = encoder.state_dict()
    # শুধুমাত্র এনকোডারের ওজনগুলো ফিল্টার করে নেওয়া হচ্ছে
    pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.replace("encoder.", "") in model_dict}
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)
    encoder.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

    print("Fine-tuning Inverse Design Model on GPU...")
    encoder.train()
    for epoch in range(30):
        total_loss = 0
        for jm, target_param in loader:
            jm, target_param = jm.to(device), target_param.to(device)
            optimizer.zero_grad()
            
            pred_param = encoder(jm)
            loss = criterion(pred_param, target_param)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Fine-tune Epoch {epoch+1}/30 | Loss: {total_loss/len(loader):.5f}")
        
    torch.save({"model": encoder.state_dict()}, os.path.join(ckpt_dir, "inverse_model_final.pt"))
    print("Fine-tuning Complete. Final Inverse Design AI Model Saved!")

if __name__ == "__main__":
    main()