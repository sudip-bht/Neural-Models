import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
def main():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(root='Data Set/train', transform=transform_train)
    model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.aux_logits = False
    model.AuxLogits = None
    criterion = nn.CrossEntropyLoss()
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9 ,weight_decay=0.9)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,batch_size=32,num_workers=16,pin_memory=True)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

    torch.save(model.state_dict(), 'Trained Model/inception3.pt')
if __name__=='__main__':
    main()
