import torch
import os
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.models as models
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
from huggingface_hub import PyTorchModelHubMixin
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = "Data Set"
train_crop_size = 224
interpolation = "bilinear"
val_crop_size = 224
val_resize_size = 224
model_name = "inception_v3"
pretrained = True
batch_size = 32
num_workers = 16
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4
lr_step_size = 30
lr_gamma = 0.1
epochs = 50
train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")
interpolation = InterpolationMode(interpolation)
criterion = torch.nn.CrossEntropyLoss()
val_crop_size = 224
val_resize_size = 224
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "Data Set"
    train_crop_size = 224
    interpolation = "bilinear"
    model_name = "inception_v3"
    pretrained = True
    batch_size = 32
    num_workers = 16
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    lr_step_size = 30
    lr_gamma = 0.1
    epochs = 50
    train_dir = os.path.join(data_path, "train")
    
    interpolation = InterpolationMode(interpolation)

    TRAIN_TRANSFORM_IMG = transforms.Compose([

    transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
    ])


    dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=TRAIN_TRANSFORM_IMG
    )
 

    

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Creating model")
    print("Num classes = ", len(dataset.classes))
    model = torchvision.models.__dict__[model_name](pretrained=pretrained)
   
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model.aux_logits = False
    model.AuxLogits = None
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    train_loss_history = []
    print("Start training")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        len_dataset = 0
    for step, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += output.shape[0] * loss.item()
        len_dataset += output.shape[0];
        if step % 10 == 0:
            print('Epoch: ', epoch, '| step : %d' % step, '| train loss : %0.4f' % loss.item() )
        epoch_loss = epoch_loss / len_dataset
        train_loss_history.append(epoch_loss)
        print('Epoch: ', epoch, '| train loss :  %0.4f' % epoch_loss )
        lr_scheduler.step()
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
    torch.save(model.state_dict(), 'Trained Model/inception3.pt')


def model_test():
    
    model = models.inception_v3()
    state_dict = torch.load('Trained Model/inception3.pt')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2) 
    model.load_state_dict(state_dict,strict=False)

    test_dir = os.path.join(data_path, "test")
    TEST_TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(val_resize_size, interpolation=interpolation),
        transforms.CenterCrop(val_crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
    ])
    dataset_test = torchvision.datasets.ImageFolder(
        test_dir,
        transform=TEST_TRANSFORM_IMG
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True
    )
    
    model.eval()
    predicted_labels = []
    ground_truth_labels = []
    
    with torch.inference_mode():
        running_loss = 0
        for step, (image, target) in enumerate(data_loader_test):
            print(step)
            image, target = image.to(device), target.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)  # Get the predicted labels
            predicted_labels.extend(predicted.cpu().numpy())  # Convert to numpy array and add to predicted labels list
            ground_truth_labels.extend(target.cpu().numpy())
            loss = criterion(output, target)
            running_loss += loss.item()
            running_loss = running_loss / len(data_loader_test)
            
    conf_matrix = confusion_matrix(predicted_labels,ground_truth_labels, )
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["Predicate Negative", "Predictate Positive"], yticklabels=["Real Negative", "Real Positive"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__=='__main__':
   model_test()