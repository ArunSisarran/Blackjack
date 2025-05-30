import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Variables
batch_size = 32
num_classes = 53
learning_rate = 0.001
num_epochs = 40

# Device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reformat images from 224 to 128
all_transfroms = transforms.Compose([transforms.Resize((128, 128)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ])

# Creating Training Dataset
train_dataset = torchvision.datasets.ImageFolder(root='../train/',
                                                 transform=all_transfroms)

# Creating Testing Dataset
test_dataset = torchvision.datasets.ImageFolder(root='../test/',
                                                transform=all_transfroms)

# Creating Validation Dataset
validation_dataset = torchvision.datasets.ImageFolder(root='../valid/',
                                                      transform=all_transfroms)

# Creating the loader objects for processing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


# CNN class
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 128x128x3 → 128x128x32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)        # → 64x64x32

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64x32 → 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        # → 32x32x64

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 32x32x64 → 32x32x128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)        # → 16x16x128

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 16x16x128 → 16x16x256
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)         # → 8x8x256

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # 8x8x256 → 8x8x512
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)         # → 4x4x512

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # 8,192 → 1024
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)   # → 53 classes

    def forward(self, x):

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Fourth convolutional block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Fifth convolutional block
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Second fully connected layer
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Final classification layer
        x = self.fc3(x)

        return x


model = ConvNeuralNet(num_classes).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

total_step = len(train_loader)

# Training
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(265, 100 * correct / total))

torch.save(model, 'card_classifier_model.pth')
print("Model saved")
