import torch
from model.diffusiondet import DiffusionDet
from dataset.tiny_coco import TinyCocoDataset
from dataset.transforms import get_test_transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model.utils import noise_boxes, cosine_beta_schedule, get_alpha_cumprod
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_boxes_for_eval(num_proposals, image_size):
    """Generate well-distributed initial boxes for evaluation"""
    boxes = []
    for _ in range(num_proposals):
        # Generate random box with minimum size
        min_size = 32
        max_size = image_size // 2
        
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        
        x = random.randint(0, image_size - w)
        y = random.randint(0, image_size - h)
        
        box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        boxes.append(box)
    return torch.stack(boxes)

def evaluate(config):
    model = DiffusionDet(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load("weights/diffusiondet.pth"))
    model.eval()

    dataset = TinyCocoDataset(num_images=10, transform=get_test_transforms(config['image_size']))

    betas = cosine_beta_schedule(config['timesteps'])
    alphas_cumprod = get_alpha_cumprod(betas).to(device)

    with torch.no_grad():
        for idx in range(5):  # visualize 5 images
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            image_size = image.shape[-1]

            # Generate well-distributed initial boxes
            noisy_boxes = sample_boxes_for_eval(config['num_proposals'], image_size)
            noisy_boxes = noisy_boxes.unsqueeze(0).to(device)
            
            # Normalize boxes
            noisy_boxes = noisy_boxes / image_size

            # Add noise
            t = torch.randint(0, config['timesteps'], (1,), device=device).long()
            noisy_boxes = noise_boxes(noisy_boxes, t, alphas_cumprod)

            pred_boxes, pred_logits = model(image, noisy_boxes)

            # Convert predictions back to image coordinates
            pred_boxes = pred_boxes.squeeze(0).cpu()
            pred_boxes = pred_boxes.clamp(0, 1)  # keep in [0,1]
            pred_boxes = pred_boxes * image_size

            # Get predicted classes
            pred_classes = torch.argmax(pred_logits.squeeze(0), dim=1).cpu()

            # Create figure
            plt.figure(figsize=(12, 8))
            plt.imshow(sample['image'].permute(1,2,0).cpu())
            
            # Plot ground truth boxes
            for box in sample['boxes']:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), 
                                       linewidth=2, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)

            # Plot predicted boxes with class colors
            class_colors = ['r', 'b', 'y']  # Red for person, Blue for car, Yellow for dog
            for box, cls in zip(pred_boxes, pred_classes):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), 
                                       linewidth=2, edgecolor=class_colors[cls], 
                                       facecolor='none', alpha=0.7)
                plt.gca().add_patch(rect)

            plt.title(f'Image {idx+1} - Green: Ground Truth, Colored: Predictions')
            plt.axis('off')
            plt.savefig(f"eval_result_{idx}.png", bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    from main import load_config
    config = load_config()
    evaluate(config)
