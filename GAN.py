#Projet DeepLearning - Dans ce script, nous allons réaliser un Gan conditionnel permettant de générer n'importe quelle nombre indiqué par l'utilisateur
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as U


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
chiffreRecrée=3

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

#Creation label personallisé
def create_labels(batchsize, target_label):
    # Labels pour la classe que vous voulez (ici, le chiffre 2)
    labels = torch.full((batchsize,), target_label, dtype=torch.long).to(device)
    one_hot_labels = F.one_hot(labels, num_classes=10).float().to(device)
    return one_hot_labels

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
            self.label_embedding = nn.Linear(10, 100)  # Embedding du label dans la même taille que l'image
            self.deconv1 = nn.ConvTranspose2d(200,256,kernel_size=7,stride=2,padding=0) #(Image 7x7 généré 256 fois) #200 car label en plus
            self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)#image 14x14 généré 128 fois
            self.deconv3 = nn.ConvTranspose2d(128,1,kernel_size=4,stride=2,padding=1) #image 28x28 généré 1 fois

            # a voir pour rajouter bachnorm

    def forward(self,noise,labels):
            # Créer un vecteur latent conditionné sur le label pour qu'il soit compatible
            label_embedding = self.label_embedding(labels).view(labels.size(0), 100, 1, 1)
            x = torch.cat([noise, label_embedding], dim=1)  # Concaténation bruit + labels
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


#3) Entrainement du discriminateur et du générateur

#creation des modeles
modelGen=Generator().to(device)
modelDiscr=CNN().to(device)

#Creation de la loss function / optimiseur
lossFunction = nn.BCELoss()
optimiseurDiscr = torch.optim.Adam(modelDiscr.parameters(), lr=0.001)
optimiseurGen = torch.optim.Adam(modelGen.parameters(), lr=0.001)

iteration = 60000

for i in range(iteration):

    if i + batchsize > len(X_train_tensor):
        break

    discrtotal_loss=0.0

     # Initialisation des gradients
    optimiseurDiscr.zero_grad()
    
    # Sélection d'un lot d'images réelles
    real_img = X_train_tensor[i * batchsize:(i + 1) * batchsize].to(device)

    #Creation des labels (pour non conditionnel)
    #real_labels = torch.ones(batchsize,1).to(device)
    #fake_labels = torch.zeros(batchsize,1).to(device)

    #Creation des labels (pour conditionnel)
    choicelabels = create_labels(batchsize,3)
    real_labels = torch.ones(batchsize,1).to(device)
    fake_labels = torch.zeros(batchsize,1).to(device)
    

    #passage dans le CNN des vrais images
    prediction_real   = modelDiscr.forwardPropagation(real_img,choicelabels)
    #On calcul la loss function uniquement sur la partie des images réels et on attribue un label reel
    loss_real = lossFunction(prediction_real,real_labels)

    #Generation d'une image fausse
    noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
    fake_img=modelGen(noise,choicelabels)
    #passage de l'image fausse dans le discriminateur et on mets le fake label dessus
    prediction_fake = modelDiscr.forwardPropagation(fake_img,choicelabels)
    loss_fake = lossFunction(prediction_fake,fake_labels)

    discrloss = loss_real+loss_fake #mesure du discriminateur pour l'optimiseur
    discrloss.backward()
    optimiseurDiscr.step()
    
    discrtotal_loss += discrloss.item()   
    print(f"iteration : {i}")  


    #entrainement generateur
    if i % 5 == 0: # on entraine le générateur tous les 5 iterations
        
        optimiseurGen.zero_grad()
        #Generation d'une image
        noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
        Gen_img=modelGen(noise,choicelabels)
        
        prediction_Gen = modelDiscr.forwardPropagation(Gen_img, choicelabels)
        gen_loss = lossFunction(prediction_Gen, real_labels)
        print(f"calcul perte gen : {gen_loss}")  
        gen_loss.backward()
        optimiseurGen.step()

        if i % 10000 == 0:  # affichage des images toutes les 100 itérations
            with torch.no_grad():
                gen_images = modelGen(noise, choicelabels).detach().cpu()
                grid = U.make_grid(gen_images, nrow=5, normalize=True)
                plt.imshow(np.transpose(grid, (1, 2, 0)))
                print(choicelabels[0])
                plt.show()





