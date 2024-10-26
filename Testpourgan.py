#Projet DeepLearning - Dans ce script, nous allons réaliser un Gan conditionnel permettant de générer n'importe quelle nombre indiqué par l'utilisateur
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as U
from torch.utils.data import DataLoader,TensorDataset

import matplotlib.pyplot as plt
from PIL import Image

# pour database d'image MNIST
import keras
from keras import layers


#Initialisation
#On defini une variable permettant de vérifier si un GPU est disponible et d'attribuer une tache à celui-ci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

numberRecreated=2
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)[:, np.newaxis, :, :] 
labelsTensor = torch.tensor(Y_train).long() 

transforme = transforms.Compose([ 
                        transforms.Normalize([.5],[.5])
                       ])

# Création du dataset avec images et labels
dataset = TensorDataset(transforme(X_train_tensor), labelsTensor)

batchsize   = 64
data_loader = DataLoader(dataset,batch_size=batchsize,shuffle=True,drop_last=True)

#Initialisation du CNN Binaire (vrai ou faux)
class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__() 
            self.label_embedding = nn.Linear(10, 28 * 28)  # Embedding du label dans la même taille que l'image
            self.conv1 = nn.Conv2d(2, 64, 5, 1, 0, bias=False) #2 cannaux car cannal + label
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 1) #modif pour avoir un résultat binaire
         
    def forwardPropagation(self,x,labels):
        label_embedding = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)
        x = torch.cat([x, label_embedding], dim=1)  # Concaténation image + label
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
            self.label_embedding = nn.Linear(10, batchsize)  # Embedding du label dans la même taille que l'image
            self.deconv1 = nn.ConvTranspose2d(101,256,kernel_size=7,stride=2,padding=0) #(Image 7x7 généré 256 fois) 101 car label en plus
            self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)#image 14x14 généré 128 fois
            self.deconv3 = nn.ConvTranspose2d(128,1,kernel_size=4,stride=2,padding=1) #image 28x28 généré 1 fois

            # a voir pour rajouter bachnorm

    def forward(self,noise,labels):
            # Créer un vecteur latent conditionné sur le label pour qu'il soit compatible
            label_embedding = self.label_embedding(labels).view(batchsize, 1, 1, 1)
            x = torch.cat([noise, label_embedding], dim=1)  # Concaténation bruit + labels
            x=F.leaky_relu(self.deconv1(x))
            y=F.leaky_relu(self.deconv2(x))
            z=torch.tanh(F.leaky_relu(self.deconv3(y))) # fonction d'activation tangente hyperbolique : centralise les valeurs entre -1 et 1, facilite l'apprentissage.
            return z
        
#3) Entrainement du discriminateur et du générateur

#creation des modeles
modelGen=Generator().to(device)
modelDiscr=CNN().to(device)

#Creation de la loss function / optimiseur
lossFunction = nn.BCELoss()
optimiseurDiscr = torch.optim.Adam(modelDiscr.parameters(), lr=0.001)
optimiseurGen = torch.optim.Adam(modelGen.parameters(), lr=0.001)



#ENTRAINEMENT

# Paramètres d'entraînement
num_epochs = 1
latent_dim = 100  # Dimension de bruit d'entrée pour le générateur
label_dim = 10    # Nombre de classes de labels (de 0 à 9 pour MNIST)
#creation du labels du nombres a obtenir en oneshot
choicelabels = torch.eye(label_dim)[numberRecreated].to(device).float() 

for epoch in range(num_epochs):
    for i, (real_img, img_labels_ori) in enumerate(data_loader):
        print(img_labels_ori)
        discrtotal_loss=0.0
        # Initialisation des gradients
        optimiseurDiscr.zero_grad()

        #on mets les données au bon format
        real_img = real_img.to(device).float()
        img_labels = torch.eye(label_dim)[img_labels_ori].to(device).float()  # Embedding du label en one-hot

        # I) Entraînement du Discriminateur
        #Creation des labels binaires
        real_labels = torch.ones(batchsize,1).to(device)
        fake_labels = torch.zeros(batchsize,1).to(device)
        # Masque pour sélectionner les labels correspondant au `numberRecreated`. mask_real vaut 1 si c'est le bon et 0 si c'est pas le bon
        mask_label = (img_labels_ori == numberRecreated).float().view(-1, 1).to(device)


        #passage dans le CNN des vrais images et loss function on fonction de l'image renseigné
        prediction_real   = modelDiscr.forwardPropagation(real_img,img_labels)

        loss_real = lossFunction(torch.where(mask_label == 1, prediction_real, 1 - prediction_real),real_img)

        #Generation d'une image fausse
        
        noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
        fake_img=modelGen(noise,choicelabels)

        


        


