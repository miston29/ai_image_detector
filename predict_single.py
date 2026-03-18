import timm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

model = timm.create_model("efficientvit_b0", pretrained=False, num_classes=1)
checkpoint = torch.load("models\\deepfake_model_final.pt", map_location="cuda")
model.load_state_dict(checkpoint)

model.eval()

transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def singlePredict(IMAGEPATH, THRESH):
    image = Image.open(IMAGEPATH).convert("RGB")
    iarr = np.array(image)
    timg = transform(image=iarr)
    itensor = timg["image"].unsqueeze(0)

    with torch.no_grad():
        logits = model(itensor)
        prob = torch.sigmoid(logits).item()

        print(f"realness : {prob*100:.2f}%          threshold : {THRESH*100}%")
        print("real" if prob > THRESH else "fake")

        if prob > THRESH:
            print(f"{IMAGEPATH} is most likely REAL")

        elif prob > THRESH*0.8:
            print(f"{IMAGEPATH} might be FAKE")

        elif prob > THRESH*0.6:
            print(f"{IMAGEPATH} IS most likely FAKE")

        else:
            print(f"{IMAGEPATH} has very high chances of being FAKE")


image_path = "custom\\im2.png"

singlePredict(image_path, 0.90)



