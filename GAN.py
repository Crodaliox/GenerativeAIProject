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

numberRecreated=3


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)[:, np.newaxis, :, :] 
labelsTensor = torch.tensor(Y_train).long() 

#on normalise entre -1 et 1 les images
transforme = transforms.Compose([ 
                        transforms.Normalize([.5],[.5])
                       ])

# Création du dataset avec images et labels
dataset = TensorDataset(transforme(X_train_tensor), labelsTensor)

batchsize   = 64
data_loader = DataLoader(dataset,batch_size=batchsize,shuffle=True,drop_last=True)
BatchNR = torch.tensor([numberRecreated]*batchsize).to(device)

#Initialisation du CNN Binaire (vrai ou faux)
class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__() 
            self.label_embedding = nn.Linear(10, 28 * 28)  # Embedding du label dans la même taille que l'image
            self.conv1 = nn.Conv2d(2, 32, 5, 1, 0, bias=False) #2 cannaux car cannal + label
            self.bn1 = nn.BatchNorm2d(32)  # Normalisation pour chaque canal :ajuste les activations pour qu'elles aient une moyenne de 0 et un écart-type de 1
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(64) 
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 1) #modif pour avoir un résultat binaire
         
    def forwardPropagation(self,x,labels):
        label_embedding = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)
        x = torch.cat([x, label_embedding], dim=1)  # Concaténation image + label
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.maxpooling1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
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
            self.label_embedding = nn.Linear(10, batchsize)  # Embedding du label dans la même taille que le noise
            self.deconv1 = nn.ConvTranspose2d(101,256,kernel_size=7,stride=2,padding=0) #(Image 7x7 généré 256 fois) 101 car label en plus
            self.bn1 = nn.BatchNorm2d(256)
            self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)#image 14x14 généré 128 fois
            self.bn2 = nn.BatchNorm2d(128)
            self.deconv3 = nn.ConvTranspose2d(128,1,kernel_size=4,stride=2,padding=1) #image 28x28 généré 1 fois

            # a voir pour rajouter bachnorm

    def forward(self,noise,labels):
            # Créer un vecteur latent conditionné sur le label pour qu'il soit compatible
            label_embedding = self.label_embedding(labels).view(batchsize, 1, 1, 1)
            x = torch.cat([noise, label_embedding], dim=1)  # Concaténation bruit + labels
            x=F.leaky_relu(self.bn1(self.deconv1(x)))
            y=F.leaky_relu(self.bn2(self.deconv2(x)))
            z=torch.tanh(F.leaky_relu(self.deconv3(y))) # fonction d'activation tangente hyperbolique : centralise les valeurs entre -1 et 1, facilite l'apprentissage.
            return z
        
#3) Entrainement du discriminateur et du générateur

#creation des modeles
modelGen=Generator().to(device)
modelDiscr=CNN().to(device)

#Creation de la loss function / optimiseur
lossFunction = nn.BCELoss()
optimiseurDiscr = torch.optim.Adam(modelDiscr.parameters(), lr=0.00001)
optimiseurGen = torch.optim.Adam(modelGen.parameters(), lr=0.00001)




#ENTRAINEMENT

# Paramètres d'entraînement
num_epochs = 5
latent_dim = 100  # Dimension de bruit d'entrée pour le générateur
label_dim = 10    # Nombre de classes de labels (de 0 à 9 pour MNIST)
#creation du labels du nombres a obtenir en oneshot
choicelabels = torch.eye(label_dim)[numberRecreated].to(device).float() 
Batchchoicelabels = torch.eye(label_dim, device=device)[BatchNR].float()
# Initialisation des listes pour stocker les pertes
losses_discr = []
losses_gen = []

for epoch in range(num_epochs):
    total_discr_loss = 0
    total_gen_loss = 0
    nbofrepetition=0
    for i, (real_img, img_labels_ori) in enumerate(data_loader):
        # Initialisation des gradients
        optimiseurDiscr.zero_grad()

        #real_img = real_img + 0.05 * torch.randn_like(real_img) #mise en place de léger bruit dans l'image reel
        

        #on mets les données au bon format
        real_img = real_img.to(device).float().to(device)
        img_labels = torch.eye(label_dim)[img_labels_ori].to(device).float()  # Embedding du label en one-hot
        

        # I) Entraînement du Discriminateur

        #Creation des labels binaires
        real_labels = torch.ones(batchsize,1).to(device)
        fake_labels = torch.zeros(batchsize,1).to(device)
        # Masque pour sélectionner les labels correspondant au `numberRecreated`. mask_real vaut 1 si c'est vrai et 0 si c'est faux
        mask_label = (img_labels_ori == numberRecreated).float().view(-1, 1).to(device)
        

        #passage dans le CNN des vrais images et loss function on fonction de l'image renseigné
        prediction_real   = modelDiscr.forwardPropagation(real_img,img_labels)
        loss_real = lossFunction(prediction_real,torch.where(mask_label == 1, real_labels, fake_labels)) #torch.where fonctionne comme un if avec le else en dernier paramètre.

        #Generation d'une image fausse
        noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
        fake_img=modelGen(noise,choicelabels)
        #passage de l'image fausse dans le discriminateur et on mets le fake label dessus
        prediction_fake = modelDiscr.forwardPropagation(fake_img,Batchchoicelabels)
        loss_fake = lossFunction(prediction_fake,fake_labels)

        #Evaluation pour l'entrainement du discriminateur
        if i % 2 == 0:  # entraîner le discriminateur seulement tous les deux pas
            discrloss = loss_real+loss_fake
            discrloss.backward()
            optimiseurDiscr.step()
            total_discr_loss += discrloss.item()


        #Entrainement generateur :
        if i % 1 == 0: # on entraine le générateur tous les  iterations
            
            #Generation d'une image
            noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
            Gen_img=modelGen(noise,choicelabels)
            
            prediction_Gen = modelDiscr.forwardPropagation(Gen_img, Batchchoicelabels)
            gen_loss = lossFunction(prediction_Gen, real_labels)
            optimiseurGen.zero_grad()
            gen_loss.backward()
            optimiseurGen.step()
            total_gen_loss += gen_loss.item()

    
    #Moyenne des pertes pour l'epoch
    avg_discr_loss = total_discr_loss / len(data_loader)
    avg_gen_loss = total_gen_loss / len(data_loader)
    losses_discr.append(avg_discr_loss)
    losses_gen.append(avg_gen_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {avg_discr_loss:.4f}, Generator Loss: {avg_gen_loss:.4f}")

    if epoch % 1 == 0:
        with torch.no_grad():
            fixed_noise = torch.randn(batchsize, 100, 1, 1).to(device)  # Générer 16 échantillons
            generated_images = modelGen(fixed_noise, choicelabels)
            generated_images = (generated_images + 1) / 2  # Ramener à l'échelle [0, 1] pour visualisation

            # Afficher les images
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for j in range(16):
                axs[j // 4, j % 4].imshow(generated_images[j].cpu().squeeze(), cmap='gray')
                axs[j // 4, j % 4].axis('off')
            plt.show()
        


        


