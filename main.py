import torch 
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FashionMNIST

import torchinfo
import csv
import os

from efficientnet import *
from resnet import *
from utils import *
from parsecmd import *

args = get_parser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 10

label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
    
# Training
def train(net, trainloader, optimizer, criterion, scheduler):

    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        loss = loss.float()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.dont_display_progress_bar==False:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                        % (train_loss/(batch_idx+1) ,  100.*correct/total, correct, total))
            
    scheduler.step()


def test(epoch, best_acc, net, testloader, criterion, save_ckpt=True, save_preds = False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs) 
            loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Necesitamos guardar las predicciones al final
            if save_preds:
                all_preds.extend(predicted.cpu().numpy().tolist())

            if args.dont_display_progress_bar==False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and save_ckpt==True:
        save_checkpoint(net, acc=acc, epoch=epoch)
        best_acc = acc

    if save_preds:
        return acc, best_acc, all_preds
    else:
        return acc, best_acc 


def get_submission_csv(preds, output_file='submission.csv'):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])  # Escribe el 'header' del csv

        # Anotamos las predicciones para las imagenes desde 60001 a 70000
        for idx, pred in enumerate(preds, start=60001):
            writer.writerow([idx, pred])

    print(f"==> Guardando resultados en: {output_file}")

def main():

    best_acc = 0
    start_epoch = 0 

    rootdir = './data/fashion_mnist'

    print('==> Preparando el dataset: Fashion MNIST')

    # Preprocesamiento del dataset y 'data augmentation' para el entrenamiento del modelo
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # Recorta la imagen de manera aleatoria
        transforms.RandomHorizontalFlip(),  # Espeja la imagen de manera aleatoria
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Valores de normalización precalculados para FashionMNIST
        transforms.RandomErasing(p= args.p, scale = (args.sh, args.sh), ratio = (args.r1, args.r1), value = [0.4914]),
        # Elimina secciones de la imagen.
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # Valores de normalización precalculados para FashionMNIST
    ])

    # FashionMNIST esta disponible como dataset dentro de 'torchvision.datasets'
    trainset = FashionMNIST(root=rootdir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = FashionMNIST(root=rootdir, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Modelo
    print('==> Construyendo modelo..')
    if args.architecture == 'ResNet':
        net = ResNet(depth=20, num_classes=NUM_CLASSES)
    elif args.architecture == 'EfficientNet':
        net = EfficientNetB0()
    else:
        print('Error: Modelo no válido')
        return

    # Muestra la arquitectura del modelo
    if args.summary:
        torchinfo.summary(net)

    net = net.to(device)
    if args.resume or args.test_model:
        ckpt = args.ckpt_file
        print('==> Cargando modelo..')
        assert os.path.isdir('checkpoint'), 'Error: no existe el checkpoint seleccionado!'
        checkpoint = torch.load(ckpt)

        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,250], gamma=0.1)

    epoch = 0
    if args.test_model==False:
        test(0, best_acc, net, testloader, criterion, save_ckpt=False)

        total_epoch=args.epochs

        for epoch in range(start_epoch, total_epoch):
            print(f"\nEpoch: {epoch} \t LR: {get_lr(optimizer): .7f}"+
                f"\t Best Acc: {best_acc:.3f}%")

            train(net, trainloader, optimizer, criterion, scheduler)
            _, best_acc = test(epoch, best_acc, net, testloader, criterion)


    print('==> Testeando al mejor modelo...')
    _, _, preds = test(epoch, best_acc, net, testloader, criterion, save_ckpt=False, save_preds=True)
    print(f"Mejor precisión: {best_acc}")

    # Obtenemos el CSV para subir a la página del Hackathon
    get_submission_csv(preds)



if __name__ == "__main__":
    main()

