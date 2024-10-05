#Projet DeepLearning
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
from PIL import Image

# pour database d'image MNIST
import keras
from keras import layers


#Initialisation
#On defini une variable permettant de vérifier si un GPU est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# On load le database d'image qui servira d'entrainement
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# X_train= image
# Y_train = etiquette : indique quel chiffre (de 0 à 9) est représenté dans chaque image.

# Normalisation des images de la database entre 0 et 1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

#Methode pour convertir les images par des tensor

X_train_tensor = torch.tensor(X_train)[:, np.newaxis, :, :] #images sous forme de tenseur à 4 dimensions ((batch_size, channels, height, width)
Y_train_tensor = torch.tensor(Y_train).long()  # .long mets chaque valeur a 64 bits
print(X_train_tensor.shape)
print(Y_train_tensor.shape)

#entrainement !

# On selectionne 10 img et 10 label
X_train_10 = X_train_tensor[:10]
Y_train_10 = Y_train_tensor[:10]

#Creation du CNN avec une class
class CNN:

    def __init__(self):
        #Dans cette methode, on définit l'ensemble des étapes qui constitue le model CNN
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 0, bias=False)
        self.maxpolling1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
        self.maxpolling2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcl1 = nn.Linear(1024, 128)
        self.fcl2 = nn.Linear(128,10)
        
    def ForwardPropagation(self,x):
        #On vient appliquer des fonctions d'activation sur chaque perceptron du model
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpolling1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpolling2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.fcl1(x))
        x = F.softmax(self.fcl2(x),dim=1)

        return x
    


#Boucle d'entrainement
#C'est la ou le model CNN va être entrainé avec des données

#Creation du model :
#model = CNN().to(device)

iteration = 5
batchSize = 64 # Les données sont séparé par lot de 64
# Créer un DataLoader pour charger les données par lots et les mélanger pour ne pas avoir toujours la même chose
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batchSize, shuffle=True)

for iteration in range(iteration):

    lossEvaluation=0.0
    allBatch = len(train_loader)  # Nombre total de batchs

    for i, (inputs, labels) in enumerate(train_loader):  # Boucle qui traverse chaque batch pour avoir toutes les données
        inputs, labels = inputs.to(device), labels.to(device)  # Envoye des données des batchs sur GPU
        
    #Creation de la loss function ou la fonction cout (où l'on va calculer l'errer)
    #Comparaison entre les etiquettes du Y_test initial à ceux sortie.

    #Pour l'instant fonction utilisé par pytorch.
    #Elle effectue automatiquement la fonction d'activation de softmax pour chaque valeur


    
    
