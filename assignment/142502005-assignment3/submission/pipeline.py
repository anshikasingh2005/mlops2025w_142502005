import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import os
import json
import tomli
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torchvision.models import (ResNet34_Weights, ResNet50_Weights,ResNet101_Weights, ResNet152_Weights)

# Load Config
with open("config.json", "r") as f:
    config = json.load(f)
with open("params.toml", "rb") as f:
    params = tomli.load(f)
with open("grid.json", "r") as f:
    grid = json.load(f)
model_dict = {
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1),
}

# Image processing as per requirement
preprocess = transforms.Compose([
    transforms.Resize(256),          
    transforms.CenterCrop(224),      
    transforms.ToTensor(),           
    transforms.Normalize(            
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
image_folder = config["data_source"]
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
for model_name in config["model"]:
    data_source = config["data_source"]
    print(f"Using {model_name} on data from {data_source}")
    
    # Load Pretrained ResNet 
    model_fn, weight_cls = model_dict[model_name]
    model = model_fn(weights=weight_cls)
    model.eval()

    # Load Hyperparameters
    base_lr = params[model_name]["learning_rate"]
    batch_size = params[model_name]["batch_size"]

    # Inference
    results = []
    class_labels = weight_cls.meta["categories"]
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)
        top5_predictions = [
            {"label": class_labels[idx], "probability": float(top5_prob[i])}
            for i, idx in enumerate(top5_catid)
        ]

        print(f"Image: {os.path.basename(img_path)}, Output shape: {output.shape}")
        for pred in top5_predictions:
            print(f"  {pred['label']}: {pred['probability']:.4f}")

        results.append({
            "model": model_name,
            "image": os.path.basename(img_path),
            "output": top5_predictions  
        })

    # Grid Search
    for lr in grid["learning_rates"]:
        for opt in grid["optimizers"]:
            for mom in grid["momentum"]:
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGB")
                    input_tensor = preprocess(img).unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output[0], dim=0)
                        top_prob, top_idx = torch.max(probs, dim=0)
                        predicted_label = class_labels[top_idx]

                    config_result = {
                        "model": model_name,
                        "image": os.path.basename(img_path),
                        "lr": lr,
                        "optimizer": opt,
                        "momentum": mom,
                        "prediction": predicted_label,
                        "confidence": float(top_prob)
                    }
                    results.append(config_result)
                    print(config_result)

    # Save Results at results.txt
    with open("results.txt", "a") as f:
        f.write(f"Using {model_name} on data from {data_source}"+ "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("The results have been saved in results.txt file")
