# Image Classification Score Prediction

This project contains a Python script that predicts the scores for all images in a specified folder. The script fuses probabilities from a efficientNet based model and another based on pretrained clip Image Encoder to generate predictions and outputs the classification scores. The generators that can be detected are StyleGAN1, StyleGAN2, StyleGAN3, Latent Diffusion, Stable Diffusion 1.5, Stable Diffusion 2.1, Stable Diffusion XL, MidJourney, Adobe FireFly (Online), Dall-e 3 (Bing Image Creator), Imagine – Meta AI, Adobe Photoshop Firefly (Gen Fill), Stable Diffusion XL Turbo or a natural image

## Requirements

To run the script, you need the following Python packages:

- `numpy==1.22.4`
- `Pillow==10.4.0`
- `torch==1.10.1+cu111`
- `torchvision==0.11.2`

## Run

You can install all required packages using `pip` with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

To run:

First download the weights from : https://drive.google.com/drive/folders/1b25mX6gvDIF_3ZW1EHWhPg4jlw0tXvwJ?usp=drive_link

then run

```bash
python predict_scores.py --folder_path /path/to/your/image_folder
```

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) - see the LICENSE file for details.