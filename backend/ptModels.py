import torch
from torchvision import transforms
from PIL import Image
import io
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from timm import create_model
from ultralytics import YOLO
from matplotlib import pyplot as plt




interpolation = InterpolationMode.BILINEAR
val_crop_size = 224


def predict_image(image_path,modelName):
    modelName=modelName.lower()
    print(modelName)
    
    if modelName=='yolov8':
        model = YOLO('yolov8n-cls.pt')  
        
    elif modelName=="inceptionv3":
        model = models.inception_v3(init_weights=True)
        state_dict = torch.load('../models/inceptionv3.pt',map_location=torch.device('cpu'))
        
        #loading the model
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2) 
        model.load_state_dict(state_dict,strict=False)
        
        val_resize_size = 299
        
    elif modelName=='resnext':
        model = torch.load("../models/resnext.pt",torch.device('cpu'))
        model.to('cpu')
        
        val_resize_size = 224
   
    elif modelName=="inceptionv4":
        model = create_model('inception_v4', pretrained=False)
        model.aux_logits = False  # Disable auxiliary logits
        model.AuxLogits = None
        model.load_state_dict(torch.load('../models/inceptionv4.pt',map_location='cpu'))
        model.to('cpu')
        
        #resize adjustment for inceptionv4
        val_resize_size = 299
       
        
    
    
    # Set the model to evaluation mode
    model.eval()
    
  
        
    try:
        # transformation to be applied to the image
        TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(val_resize_size, interpolation=interpolation),
        transforms.CenterCrop(val_crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )])
        
     
        
        
        # load image
        image_not_transformed = Image.open(image_path).convert('RGB')
        
        # transforming the image
        image = TRANSFORM_IMG(image_not_transformed).unsqueeze(0).to('cpu')  
        
        
        
        # predictions
        with torch.no_grad():
            if modelName=="resnext":
                output = torch.sigmoid(model(image))
                prediction = 1 if output.item() >= 0.5 else 0
            
            elif modelName=="yolov8":
                prediction = model(image_not_transformed)
            else:
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = predicted.item()
                
        print(f"Prediction: {prediction}")
            
       
        
        # Interpret the prediction 
        if prediction == 0:
            return "benign"  
        elif prediction == 1:
            return "malignant" 
        else:
            return "unknown class"  
    
    
    except Exception as e:
        print("Prediction Error:", e)
        return None

if __name__ == '__main__':
    image_path = './output_image.png'  
    result = predict_image(image_path,'yolov8')
    print(f"Result: {result}")