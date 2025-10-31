import os
import json
import argparse
import urllib.request
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

# === CONFIG ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 50  # Smaller batch size for small dataset
EPOCHS = 100
LEARNING_RATE = 1e-4  # Higher learning rate for faster convergence
VAL_SPLIT = 0.2
PATIENCE = 5  # Early stopping patience

# === UTILS: PIL loader with white background ===
def pil_loader_white_bg(path):
    img = Image.open(path).convert("L")  # Convert to grayscale
    # Binarize: everything not white becomes black
    arr = np.array(img)
    # Threshold: set all pixels < 250 to 0 (black), else 255 (white)
    arr = np.where(arr < 250, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    return img


# === DOWNLOAD .NDJSON FILE ===
def download_ndjson(class_name, out_dir="quickdraw"):
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{class_name}.ndjson"
    dest = os.path.join(out_dir, f"{class_name}.ndjson")
    if not os.path.exists(dest):
        print(f"Downloading {class_name}.ndjson...")
        urllib.request.urlretrieve(url, dest)
    return dest

# === DRAW .NDJSON DOODLES ===
def draw_ndjson_to_png(ndjson_path, out_dir, max_count=1000, prefix=""):
    drawing = json.loads(open(ndjson_path).readline())
    print(drawing['word'], len(drawing['drawing']))
    for stroke in drawing['drawing']:
        plt.plot(stroke[0], stroke[1])
    plt.gca().invert_yaxis()
    plt.show()

def draw_ndjson_to_pngs(ndjson_path, out_dir, max_count=1000, prefix=""):
    start = 10000
    rows, cols = 4, 10
    subplot_w, subplot_h = 256, 256  # pixels
    dpi = 100 
    fig_w = (subplot_w * cols) / dpi
    fig_h = (subplot_h * rows) / dpi
    max_count = rows * cols
    #fig, axes = plt.subplots(row, col, figsize=(10, 4))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = axes.flatten()
    with open(ndjson_path, 'r') as f:
        lines = f.readlines()    
        line_count = sum(1 for _ in f)
        print("File lines:", len(lines))
    for i, line in enumerate(lines, start):
        #if i < start: 
        #    continue
        drawing = json.loads(line)
        axes_index = i - start
        axes[axes_index].set_title(f"{drawing['word']}-{len(drawing['drawing'])}")
        for stroke in drawing['drawing']:
            axes[axes_index].plot(stroke[0], stroke[1])
            #plt.plot(stroke[0], stroke[1])
        if axes_index >= max_count - 1: 
            break
               
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset', help='Path to dataset dir')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain with QuickDraw')
    args = parser.parse_args()

    if args.pretrain:
        print("🔧 Generating QuickDraw dataset...")

        # Fish
        fish_ndjson = download_ndjson("fish")
        draw_ndjson_to_png(fish_ndjson, os.path.join(args.data, "fish"), max_count=1950, prefix="fish_ndjson_")

        # Not fish: using multiple unrelated classes
        not_fish_classes = [
            "cat", "banana", "submarine", "face", "octopus",
            "crab", "broccoli", "cloud", "truck", "basket"
        ]
    else:
        fish_ndjson = "./quickdraw/fish.ndjson"
        draw_ndjson_to_pngs(fish_ndjson, os.path.join(args.data, "fish"), max_count=1950, prefix="fish_ndjson_")
    print("Loading dataset...")
    print("Saving model to fish_doodle_classifier.pth...")
if __name__ == "__main__":
    main()
