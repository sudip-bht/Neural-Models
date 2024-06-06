import torch
from torchvision import transforms
from PIL import Image
import io
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from timm import create_model


interpolation = InterpolationMode.BILINEAR
val_crop_size = 224
val_resize_size = 224

def predict_image(image_path,modelName):
    modelName=modelName.lower()
    print(modelName)
    
    if modelName=='inception3':
        model = models.inception_v3()
        state_dict = torch.load('Trained Model/inceptionv3.pt')
        
    elif modelName=="inceptionv3":
        model = models.inception_v3()
        state_dict = torch.load('Trained Model/inceptionv3.pt',map_location=torch.device('cpu'))
        
        #loading the model
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2) 
        model.load_state_dict(state_dict,strict=False)
        
       
        
    elif modelName=="inceptionv4":
        model = create_model('inception_v4', pretrained=False)
        model.aux_logits = False  # Disable auxiliary logits
        model.AuxLogits = None
        model.load_state_dict(torch.load('Trained Model/inceptionv4.pt',map_location='cpu'))
        model.to('cpu')
       
        
    
    
    # Set the model to evaluation mode
    model.eval()
    
    print("before try section")
        
    try:
        # Define the transformation to be applied to the image
        TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(val_resize_size, interpolation=interpolation),
        transforms.CenterCrop(val_crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )])
        
        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transformations
        image = TRANSFORM_IMG(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        
        prediction = predicted.item()
        print(f"Prediction: {prediction}")
        
        # Interpret the prediction based on your class labels
        if prediction == 0:
            return "benign"  # or any label you defined for class 0
        elif prediction == 1:
            return "malignant"  # or any label you defined for class 1
        else:
            return "unknown class"  # for any unexpected class index
    
    
    except Exception as e:
        print("Prediction Error:", e)
        return None

if __name__ == '__main__':
    image_path = 'Data Set/train/malignant/ISIC_0033700.jpg'  # Replace with the actual path to your image
    result = predict_image(image_path,"inceptionv3")
    print(f"Result: {result}")