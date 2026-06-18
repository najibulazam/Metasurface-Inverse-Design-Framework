import os
import argparse
import logging
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.simmim import build_simmim
from train.dataset_loader import load_pretrain_dataloaders

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    logger = logging.getLogger("pretrain")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "pretrain.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    logger = setup_logger()
    
    # জিপিইউ ফোর্সড সিলেক্ট এবং অপ্টিমাইজেশন ট্রিকস
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using Super-Fast GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Warning: GPU not found, using CPU.")

    train_loader, val_loader = load_pretrain_dataloaders(
        DATASET_DIR, batch_size=args.batch_size, val_ratio=0.1, mask_type=0, num_workers=4
    )

    # config অনুযায়ী কাস্টম মেটাসার্ফেস ViT মডেল বিল্ড
    from model.vit_small import VisionTransformerSmall
    encoder = VisionTransformerSmall(
        img_size=(21, 6), patch_size=1, in_chans=1, embed_dim=512, depth=8, num_heads=16, num_para=0
    )
    from model.simmim import build_simmim
    model = build_simmim(encoder, loss_type=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda") # Mixed precision for RTX 5070 Ti

    logger.info("GPU Training Started...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        
        for img, mask in train_loader:
            img = img.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss, _ = model(img, mask)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        # ভ্যালিডেশন লুপ
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device, non_blocking=True).float()
                mask = mask.to(device, non_blocking=True).float()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss, _ = model(img, mask)
                val_loss_total += loss.item()
        
        val_loss = val_loss_total / max(1, len(val_loader))
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {elapsed:.1f}s")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(CKPT_DIR, f"simmim_epoch{epoch+1}.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)
            logger.info(f"Saved Checkpoint to: {ckpt_path}")

if __name__ == "__main__":
    main()