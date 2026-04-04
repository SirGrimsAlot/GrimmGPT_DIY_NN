import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import dataset as ImageDataSet
import neural_net as NetModel


data_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#Load dataset
trainDataset = ImageDataSet(csv_file='train_labels.csv', root_dir='train_set', transform=data_transform)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)

num_classes = len(trainDataset.annotations.iloc[:, 1].unique())
model = NetModel(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train loop

for epoch in range(20):
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/20] done.")