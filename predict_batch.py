import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import numpy as np
import timm
import os

transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

model = timm.create_model("efficientvit_b0", pretrained=False, num_classes=1)
checkpoint = torch.load("models\\deepfake_model_final.pt", map_location="cuda")
model.load_state_dict(checkpoint)

model.eval()


def batchPredict(FOLDERPATH, THRESH):
    for image in os.listdir(FOLDERPATH):
        img = Image.open(f"{FOLDERPATH}\\{image}").convert("RGB")
        iarr = np.array(img)
        timg = transform(image=iarr)
        itensor = timg["image"].unsqueeze(0)


        with torch.no_grad():
            logits = model(itensor)
            prob = torch.sigmoid(logits).item()

            if prob > THRESH:
                print(f"{image}\t: REAL {prob*100:.2f}")

            else:
                print(f"{image}\t: FAKE {(1-prob)*100:.2f}")
