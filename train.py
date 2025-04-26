import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.diffusiondet import DiffusionDet
from dataset.tiny_coco import TinyCocoDataset
from dataset.transforms import get_train_transforms, get_test_transforms
from model.utils import cosine_beta_schedule, get_alpha_cumprod, noise_boxes
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_boxes(gt_boxes, num_proposals, image_size):
    """Improved box sampling with better spatial distribution"""
    boxes = []
    for boxes_list in gt_boxes:
        N = boxes_list.shape[0]
        
        # If we have enough ground truth boxes, use them
        if N >= num_proposals:
            # Randomly sample from ground truth boxes
            idx = torch.randperm(N)[:num_proposals]
            sampled = boxes_list[idx]
        else:
            # Mix ground truth boxes with random proposals
            sampled = []
            
            # Add all ground truth boxes
            sampled.extend(boxes_list)
            
            # Add random boxes with better distribution
            remaining = num_proposals - N
            for _ in range(remaining):
                # Generate random box with minimum size
                min_size = 32  # Minimum box size
                max_size = image_size // 2  # Maximum box size
                
                # Random size
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                
                # Random position ensuring box stays within image
                x = random.randint(0, image_size - w)
                y = random.randint(0, image_size - h)
                
                box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
                sampled.append(box)
            
            sampled = torch.stack(sampled)
        
        boxes.append(sampled)
    return torch.stack(boxes)

def train(config):
    # Create train and validation datasets
    train_dataset = TinyCocoDataset(
        root=config['dataset_root'],
        num_images=config['num_images'],
        transform=get_train_transforms(config['image_size']),
        is_train=True,
        test_split_ratio=config['test_split_ratio']
    )
    
    val_dataset = TinyCocoDataset(
        root=config['dataset_root'],
        num_images=config['num_images'],
        transform=get_test_transforms(config['image_size']),
        is_train=False,
        test_split_ratio=config['test_split_ratio']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=lambda x: x)

    model = DiffusionDet(num_classes=config['num_classes']).to(device)
    lr = float(config['lr'])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    betas = cosine_beta_schedule(config['timesteps'])
    alphas_cumprod = get_alpha_cumprod(betas).to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        for batch in train_pbar:
            images = torch.stack([b['image'] for b in batch]).to(device)
            gt_boxes = [b['boxes'] for b in batch]

            # Use improved box sampling
            noisy_boxes = sample_boxes(gt_boxes, config['num_proposals'], images.shape[-1])
            noisy_boxes = noisy_boxes.to(device)

            # Normalize boxes to [0,1]
            noisy_boxes = noisy_boxes / images.shape[-1]

            # Add diffusion noise with varying intensity
            t = torch.randint(0, config['timesteps'], (1,), device=device).long()
            noisy_boxes = noise_boxes(noisy_boxes, t, alphas_cumprod)

            optimizer.zero_grad()
            pred_boxes, pred_logits = model(images, noisy_boxes)

            # Add box size regularization
            pred_boxes_unnorm = pred_boxes * images.shape[-1]
            box_sizes = (pred_boxes_unnorm[:, :, 2:] - pred_boxes_unnorm[:, :, :2])
            size_loss = torch.mean(torch.relu(32 - box_sizes))  # Penalize boxes smaller than 32 pixels
            
            # Combine losses
            loss = criterion(pred_boxes, noisy_boxes) + 0.1 * size_loss
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                images = torch.stack([b['image'] for b in batch]).to(device)
                gt_boxes = [b['boxes'] for b in batch]

                noisy_boxes = []
                for boxes in gt_boxes:
                    N = boxes.shape[0]
                    if N >= config['num_proposals']:
                        idx = torch.randperm(N)[:config['num_proposals']]
                        sampled = boxes[idx]
                    else:
                        pad = config['num_proposals'] - N
                        sampled = torch.cat([boxes, torch.randn(pad, 4) * images.shape[-1]], dim=0)
                    noisy_boxes.append(sampled)
                noisy_boxes = torch.stack(noisy_boxes).to(device)

                noisy_boxes = noisy_boxes / images.shape[-1]

                t = torch.randint(0, config['timesteps'], (1,), device=device).long()
                noisy_boxes = noise_boxes(noisy_boxes, t, alphas_cumprod)

                pred_boxes, pred_logits = model(images, noisy_boxes)
                loss = criterion(pred_boxes, noisy_boxes)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/diffusiondet_best.pth")

    # Save final model
    torch.save(model.state_dict(), "weights/diffusiondet_final.pth")

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curves.png")
    plt.close()
