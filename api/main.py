from fastapi import FastAPI, UploadFile, File
import numpy as np
from trism import TritonModel
from torchvision import transforms
from PIL import Image
import io


app = FastAPI()

model = TritonModel(
  model="densenet",     # Model name.
  version=1,            # Model version.
  url="localhost:8001", # Triton Server URL. 
  grpc=True             # Use gRPC or Http.
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_img = transform(image).unsqueeze(0).numpy()
    
    outputs = model.run(data = [transformed_img])
    inference_output = outputs['fc6_1'].astype(str).tolist()
    return {"prediction": inference_output}  
