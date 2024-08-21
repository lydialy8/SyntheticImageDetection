import os
import argparse
from PIL import Image
import torch
from torchvision import models, transforms
from torch import jit
import torch.nn.functional as F
import numpy as np


def main():
    args = extract_args()
    model_eff = jit.load('eff_modelPLO.zip')
    center = jit.load('eff_modelPLO_center.zip')
    model_clip = jit.load('clip_model.zip')
    model_eff.eval()
    model_clip.eval()

    transform_clip = transforms.Compose([transforms.Resize((224, 224))])

    transform_eff = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Iterate through all images in the folder
    for filename in os.listdir(args.folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(args.folder_path, filename)
            image = load_image(image_path, transform_eff)
            scores = get_scores(image, transform_clip, center, model_eff, model_clip)
            # Print or process the scores as needed
            print(f"Scores for {filename}: {scores}")


def load_image(image_path, transform_eff):
    image = Image.open(image_path).convert('RGB')
    image = transform_eff(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def get_scores(img, transform_clip, center, model_eff, model_clip):
    with torch.no_grad():
        _, logits = model_eff(img)
        prediction = torch.sigmoid(center(logits)).item()  # the output is a scalar probability
        img = transform_clip(img)
        logits2 = model_clip(img)
        prediction2 = torch.softmax(logits2, dim=1)
    probabilities = np.maximum(prediction, prediction2[0][1].numpy())
    return probabilities


def extract_args():
    parser = argparse.ArgumentParser(description="Classify if images have evidence being synthetic")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing images.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
