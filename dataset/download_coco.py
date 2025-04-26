import os
import random
import json
from pycocotools.coco import COCO
import requests
from tqdm import tqdm

def download_coco_mini(output_dir="dataset", num_images=100):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Download COCO 2017 validation annotations
    ann_file = "instances_val2017.json"
    if not os.path.exists(ann_file):
        print("Downloading COCO 2017 annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        os.system(f"wget {url} && unzip annotations_trainval2017.zip")

    coco = COCO('annotations/instances_val2017.json')

    # Categories: person (1), car (3), dog (18)
    target_cats = ['person', 'car', 'dog']
    cat_ids = coco.getCatIds(catNms=target_cats)

    img_ids = []
    for cat_id in cat_ids:
        img_ids += coco.getImgIds(catIds=[cat_id])

    img_ids = list(set(img_ids))
    selected_imgs = random.sample(img_ids, num_images)

    # Save annotations
    final_anns = []

    for img_id in tqdm(selected_imgs):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        url = img_info['coco_url']
        img_path = os.path.join(output_dir, "images", file_name)

        # Download image
        if not os.path.exists(img_path):
            img_data = requests.get(url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            bbox = ann['bbox']  # [x,y,w,h]
            final_anns.append({
                "image": file_name,
                "bbox": bbox,
                "category_id": ann['category_id']
            })

    # Save final annotation file
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(final_anns, f, indent=4)

    print("COCO-mini dataset prepared!")

if __name__ == "__main__":
    download_coco_mini() 