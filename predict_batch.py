import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import numpy as np
import timm
from random import shuffle

SIZE = 8
ipaths = []
for i in range(SIZE):
    num = random.randint(0, 5400)
    ipaths.append((f"Deeta\\Test\\Fake\\fake_{num}.jpg", "FAKE"))

for i in range(SIZE):
    num = random.randint(0, 5400)
    ipaths.append((f"Deeta\\Test\\Real\\real_{num}.jpg", "REAL"))



transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

data_transformed = []


for imgpth, label in ipaths:
    image = Image.open(imgpth).convert("RGB")
    iarr = np.array(image)
    timg = transform(image=iarr)
    itensor = timg["image"].unsqueeze(0)

    data_transformed.append((itensor, label))

shuffle(data_transformed)

model = timm.create_model("efficientvit_b0", pretrained=False, num_classes=1)
checkpoint = torch.load("models\\deepfake_model_final.pt", map_location="cuda")
model.load_state_dict(checkpoint)

model.eval()

with torch.no_grad():
    print(f"{'ACTUAL':<{10}} {'PROBABILITY':<{15}} {'PREDICTED':<{10}} {'RESULT':<{10}}")
    print("-"*45)
    for IMAGE, LABEL in data_transformed:
        logits = model(IMAGE)
        prob = torch.sigmoid(logits).item()
        answer = "REAL" if prob > 0.90 else "FAKE"
        result = "✔" if answer == LABEL else "❌"
        print(f"{LABEL:<{10}} {prob:<{15}.3f} {answer:<{10}} {result:<{10}}")