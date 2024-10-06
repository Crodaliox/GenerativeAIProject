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

batchsize=1 #Nb d'echantillons traité en même temps

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
          z=torch.tanh(F.leaky_relu(self.deconv3(y))) # fonction d'activation tangente hyperbolique : centralise les valeurs entre -1 et 1, facilite l'apprentissage.
          return z
        

# gen=Generator()  
# noise = torch.randn(batchsize,100,1,1) # 64 échantillons, 100 dimensions, 1x1 (h, w)
# print(noise.size())
# result,result2,result3 = gen(noise)  

# #affichage
# fig, axs = plt.subplots(1, 6, figsize=(8, 12))
# fig, axs2 = plt.subplots(1, 6, figsize=(8, 12))
# fig, axs3 = plt.subplots(1, 1, figsize=(8, 12))
# for i in range(6):
#     axs[i].imshow(result[0][i].detach().numpy(), cmap='gray')
#     axs[i].set_title(f'Image')
#     axs[i].axis('off')
# for i in range(6):
#     axs2[i].imshow(result2[0][i].detach().numpy(), cmap='gray')
#     axs2[i].set_title(f'Image')
#     axs2[i].axis('off')


#     axs3.imshow(result3[0][0].detach().numpy(), cmap='gray')
#     axs3.set_title(f'Image')
#     axs3.axis('off')


#3) Entrainement du discriminateur

#creation des modeles
modelGen=Generator().to(device)
modelDiscr=CNN().to(device)

#Creation de la loss function
lossFunction = nn.BCELoss()
optimiseurDiscr = torch.optim.Adam(modelDiscr.parameters(), lr=0.001)

iterationDiscr = 1000

for i in range(iterationDiscr):

    if i + batchsize > len(X_train_tensor):
        break
    
    total_loss=0.0

     # Initialisation des gradients
    optimiseurDiscr.zero_grad()
    
    # Sélection d'un lot d'images réelles
    real_img = X_train_tensor[i * batchsize:(i + 1) * batchsize].to(device)

    #Creation des labels 
    real_labels = torch.ones(batchsize,1).to(device)
    fake_labels = torch.zeros(batchsize,1).to(device)

    #passage dans le CNN des vrais images
    prediction_real   = modelDiscr.forwardPropagation(real_img)
    #On calcul la loss function uniquement sur la partie des images réels et on attribue un label reel
    loss_real = lossFunction(prediction_real,real_labels)

    #Generation d'une image fausse
    noise = noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
    fake_img=modelGen(noise)
    #passage de l'image fausse dans le discriminateur et on mets le fake label dessus
    prediction_fake = modelDiscr.forwardPropagation(fake_img)
    loss_fake = lossFunction(prediction_fake,fake_labels)

    discrloss = loss_real+loss_fake #mesure du discriminateur pour l'optimiseur
    discrloss.backward()
    optimiseurDiscr.step()

    total_loss += discrloss.item() 
    print(total_loss)
    
    



