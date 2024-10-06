#Projet DeepLearning - Dans ce script, nous allons réaliser un Gan conditionnel permettant de générer n'importe quelle nombre indiqué par l'utilisateur
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

#Initialisation du CNN Binaire (vrai ou faux)
class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__() 
            self.conv1 = nn.Conv2d(1, 64, 5, 1, 0, bias=False)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 1) #modif pour avoir un résultat binaire
         
    def forwardPropagation(self,x):
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpooling1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpooling2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.fcl1(x))
        print(torch.sigmoid(self.fcl2(x)))
        x = torch.sigmoid(self.fcl2(x))
        
        return x



#2) creation du réseau Générateur
  # Dimensionnement du vecteur bruit
#Generation d'un 1
class Generator(nn.Module):
     
    def __init__(self):
            super(Generator, self).__init__()
            #deconvolution
            self.deconv1 = nn.ConvTranspose2d(100,128,kernel_size=7,stride=2,padding=0) #(Image 7x7 généré 128 fois)
            self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)#image 14x14 généré 64 fois
            self.deconv3 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1) #image 28x28 généré 1 fois

            # a voir pour rajouter bachnorm

    def forward(self,x):
          x=F.leaky_relu(self.deconv1(x))
          y=F.leaky_relu(self.deconv2(x))
          z=F.leaky_relu(self.deconv3(y))
          print(x.size())
          print(y.size())
          print(z.size())
          return x,y,z
        

gen=Generator()  
noise = torch.randn(1,100,1,1) # 64 échantillons, 100 dimensions, 1x1 (h, w)
print(noise.size())
result,result2,result3 = gen(noise)  

#affichage
fig, axs = plt.subplots(1, 6, figsize=(8, 12))
fig, axs2 = plt.subplots(1, 6, figsize=(8, 12))
fig, axs3 = plt.subplots(1, 1, figsize=(8, 12))
for i in range(6):
    axs[i].imshow(result[0][i].detach().numpy(), cmap='gray')
    axs[i].set_title(f'Image')
    axs[i].axis('off')
for i in range(6):
    axs2[i].imshow(result2[0][i].detach().numpy(), cmap='gray')
    axs2[i].set_title(f'Image')
    axs2[i].axis('off')


    axs3.imshow(result3[0][0].detach().numpy(), cmap='gray')
    axs3.set_title(f'Image')
    axs3.axis('off')

gen=Generator()  
noise = torch.randn(1,100,1,1) # 64 échantillons, 100 dimensions, 1x1 (h, w)
print(noise.size())
result,result2,result3 = gen(noise)  

#affichage
fig, axs = plt.subplots(1, 6, figsize=(8, 12))
fig, axs2 = plt.subplots(1, 6, figsize=(8, 12))
fig, axs3 = plt.subplots(1, 1, figsize=(8, 12))
for i in range(6):
    axs[i].imshow(result[0][i].detach().numpy(), cmap='gray')
    axs[i].set_title(f'Image')
    axs[i].axis('off')
for i in range(6):
    axs2[i].imshow(result2[0][i].detach().numpy(), cmap='gray')
    axs2[i].set_title(f'Image')
    axs2[i].axis('off')


    axs3.imshow(result3[0][0].detach().numpy(), cmap='gray')
    axs3.set_title(f'Image')
    axs3.axis('off')

plt.show()

#3) Entrainement du discriminateur






