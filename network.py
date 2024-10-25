# Erica Shepherd
# CS 7180 Section 2
# Final Project

# import statements
import sys
import torch
from torch import nn
import torchvision
from torchvision.models import resnet as rn
import database as db
import numpy as np

class SEResNet_Module(nn.Module):
    '''
    creates the SE-ResNEt module structure to be used in SE-ResNet CNN
    credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, c, r=16):
        super().__init__()
        # global average pooling
        # reduces dimensions
        self.squeeze = nn.AdaptiveAvgPool2d(1) 

        # bottleneck architecture with two fully connected layers
        # increases dimensions back to original
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    # forward pass for the module
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    """
    updated BasicBlock for ResNet from torchvision to include SE module
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = rn.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = rn.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        # added SE module
        self.se = SEResNet_Module(planes, r)

    # forward pass for block
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # added SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    """
    updated Bottleneck architecture from torchvision to include SE module
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBottleneck, self).__init__()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # added SE module
        self.se = SEResNet_Module(planes * self.expansion, r)
        
        self.downsample = downsample
        self.stride = stride

    # forward pass for bottleneck
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # added SE module
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def se_resnet34(num_classes=1000):
    """
    constructs SE-ResNet-34 model with given number of classes
    :params: number of classes
    :returns: model
    """
    model = rn.ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet18(num_classes=1000):
    """
    constructs SE-ResNet-18 model with given number of classes
    :params: number of classes
    :returns: model
    """
    model = rn.ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet50(num_classes=1000):
    """
    constructs SE-ResNet-50 model with given number of classes
    :params: number of classes
    :returns: model
    """
    model = rn.ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet101(num_classes=1000):
    """
    constructs SE-ResNet-101 model with given number of classes
    :params: number of classes
    :returns: model
    """
    model = rn.ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def train_epoch(device, network, optimizer, train_loader):
    """
    trains the model for 1 epoch based on given parameters and prints process
    :params: device to be used, network model, optimization technique, epoch number,
            log interval, and training data loader
    """
    network.train() # set network to train mode
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        # feeds data and targets to cpu/gpu
        data = data.to(device)
        target = target.to(device)

        output = network(data) # run the data (forward pass) -> output
        loss = criterion(output, target) # cross entropy loss
        optimizer.zero_grad() # sets gradients to zero
        loss.backward() # execute backward pass
        optimizer.step() # step optimizer

        # prints training progress
        print('Training Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    return

def test(device, network, test_loader):
    """
    calculates losses and accuracy of network for given data loader
    :params: device to be used, network model, testing data loader
    :returns: accuracy rate of network
    """
    network.eval() # enables evaluation mode

    correct = 0
    with torch.no_grad(): # doesn't compute gradients
        for data, target in test_loader:
            output = network(data) # run the data -> output
            pred = output.argmax(dim=-1) # compare prediction with label
            correct += pred.squeeze().eq(target).sum().item() # store accuracy

    # prints result
    print('Test set: Accuracy = {}/{} ({:.0f}%)'.format(
                                correct, len(test_loader.dataset),
                                100. * correct / len(test_loader.dataset)))

    # returns accuracy rate
    return 100. * correct / len(test_loader.dataset)

def save_results(model_name, epoch, accuracy):
    """
    saves accuracies and times for given model names by writing them to textfile
    :params: network model name, accuracy of network, time taken to train network
    """
    fp = open("results.txt", "a")
    fp.write("Model: {}, Epoch: {}, Accuracy: {}\n".format(model_name, epoch, accuracy))
    fp.close()
    print("Results saved.\n")

def train_network(device, network, optimizer, n_epochs, train_loader, test_loader, models, choice):
    """
    trains network for given number of epochs and reports accuracy 
    :params: device to be used, scheduler for optimizer, optimizer, network model, epoch number, log interval,
            training data loader, and testing data loader
    :returns: final accuracy of network in test
    """
    for epoch in range(1, n_epochs+1):
        train_epoch(device, network, optimizer, train_loader)
        accuracy = test(device, network, test_loader)
        save_results(models[choice-1], str(epoch), str(accuracy))

    # returns final accuracy
    return accuracy

def main(argv):
    torch.manual_seed(0) # makes network code repeatable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # sets device
    loop = False # default value

    # model names
    models = ["SEResNet-18", "SEResNet-34", "SEResNet-50", "SEResNet-101"]
    
    # number of classes for model
    num_classes = 2

    # check for user input
    if len(sys.argv) > 1 and len(sys.argv) < 4:
        # assign the model and epoch number according to user input
        choice = int(sys.argv[1])
        epochs = int(sys.argv[2])

        if choice == 1:
            model = se_resnet18(num_classes)
        elif choice == 2:
            model = se_resnet34(num_classes)
        elif choice == 3:
            model = se_resnet50(num_classes)
        elif choice == 4:
            model = se_resnet101(num_classes)
        elif choice == 5:
            model_set = [se_resnet18(num_classes), se_resnet34(num_classes),
                            se_resnet50(num_classes), se_resnet101(num_classes)]
            loop = True
        else:
            print("Invalid model number. Please select a number between 1 and 5.")
            exit(-1)
    else:
        print("Invalid input - Please enter two numbers in [model] [epoch] format.")
        print("Model numbers: \n[1] {} \n[2] {} \n[3] {} \n[4] {}\n[5] All".format(models[0], 
                                                                models[1], models[2], models[3]))
        exit(-2)

    if torch.cuda.is_available():
        model.cuda()

    # filepaths
    training_labels = "mini_train_labels.csv"
    testing_labels = "mini_test_labels.csv"
    img_dir = "mini_dataset_128/"

    # dataset creation
    training_data = db.PneumoniaDataset(training_labels, img_dir, transform=
                                                                    torchvision.transforms.Compose([
                                                                    torchvision.transforms.ToPILImage(),
                                                                    torchvision.transforms.ToTensor()
                                                                    ]),)
    testing_data = db.PneumoniaDataset(testing_labels, img_dir, transform=
                                                                    torchvision.transforms.Compose([
                                                                    torchvision.transforms.ToPILImage(),
                                                                    torchvision.transforms.ToTensor()
                                                                    ]))

    # data loaders
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=False)
        

    if (not loop):
        # SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
        train_network(device, model, optimizer, epochs, train_dataloader, test_dataloader, models, choice)
    else:
        # runs through all four models
        choice = 1
        for model in model_set:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0001)
            train_network(device, model, optimizer, epochs, train_dataloader, test_dataloader, models, choice)
            choice += 1
        
    return

# runs code only if in file
if __name__ == "__main__":
    main(sys.argv)