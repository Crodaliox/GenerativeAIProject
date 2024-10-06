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
#On defini une variable permettant de vérifier si un GPU est disponible et d'attribuer une tache à celui-ci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#1) Creation du réseau Discriminateur
#Voir algo NumberDetection pour plus de details sur comment il fonctionne

#Load de la database
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# Normalisation
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
#Convertion en Tensor
X_train_tensor = torch.tensor(X_train)[:, np.newaxis, :, :] 
Y_train_tensor = torch.tensor(Y_train).long() 

#Initialisation du CNN
class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__() 
            self.conv1 = nn.Conv2d(1, 64, 5, 1, 0, bias=False)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 10)
         
    def forwardPropagation(self,x):
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpooling1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpooling2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.fcl1(x))
        x = F.softmax(self.fcl2(x),dim=1)
        return x

#Boucle d'entrainement
#Creation du model :
model = CNN().to(device)
#Loss Function
lossFunction = nn.CrossEntropyLoss() 
#descente de gradient
grad = torch.optim.Adam(model.parameters(), lr=0.001)

iteration = 5000
X_train_10 = X_train_tensor[:iteration].to(device)
Y_train_10 = Y_train_tensor[:iteration].to(device)

for i in range(iteration):

    total_loss=0.0
    lossEvaluation=0.0
    # Initialisation des gradients
    grad.zero_grad() 
    CNNoutput = model.forwardPropagation(X_train_10[i].unsqueeze(0))
    #On calcul le niveau d'erreur à la sortie avec la loss Function
    lossValue = lossFunction(CNNoutput,Y_train_10[i].unsqueeze(0))
    #Backward Propagation
    lossValue.backward()
    #descente de gradient
    grad.step()

    #Calcul de la moyenne d'erreur pour chaque iteration
    total_loss += lossValue.item()
    average_loss = total_loss / (i + 1)
    print(f"Iteration {i + 1}/{iteration}, Erreur moyenne : {average_loss}")

    #Class predicte pour chaque iteration
    predicted_class = torch.max(CNNoutput, 1)
    print("Classe prédicte : ", predicted_class.indices.cpu().numpy())




