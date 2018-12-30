from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
data_dir = os.environ['TESTDATADIR']
assert data_dir is not None, "No data directory"

from models import *
checkpoint = torch.load('../mobileNetV2.cifar10.pth')
model = MobileNetV2()
model.load_state_dict(checkpoint['net'])

@benchmarking(team=4, task=0, model=model, preprocess_fn=None)
def inference(model, testloader,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    model.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc

'''
TESTDATADIR="/tmp/work/data/CINIC-10/test/" python score.py
'''

if __name__=='__main__':
    transform_test = transforms.Compose([
        transforms.Resize((32,32),),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],  std=[0.24205776, 0.23828046, 0.25874835]),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    #testset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    inference(model, testloader)
