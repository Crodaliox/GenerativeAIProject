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
            self.label_embeddingBin = nn.Linear(1, 28 * 28)  # Embedding du label dans la même taille que l'image
            self.conv1 = nn.Conv2d(2, 64, 5, 1, 0, bias=False) #2 cannaux car cannal + label
            self.conv11Label = nn.Conv2d(1, 64, 5, 1, 0, bias=False) #1 cannal
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 1) #modif pour avoir un résultat binaire
            self.fclTrain = nn.Linear(128,11) # résultat pour training
         
    def forwardPropagation(self,x,labels,Disctraining,genTrain):
        if Disctraining == False:
            label_embedding = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)
            x = torch.cat([x, label_embedding], dim=1)  # Concaténation image + label
            x = F.leaky_relu(self.conv1(x))
        else:
             x = F.leaky_relu(self.conv11Label(x))
            
        
        x = self.maxpooling1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpooling2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.fcl1(x))
        if Disctraining == True and genTrain==False:
            x = F.softmax(self.fclTrain(x),dim=1)
        else:
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
lossFunctionCondit = nn.CrossEntropyLoss()
optimiseurDiscr = torch.optim.Adam(modelDiscr.parameters(), lr=0.001)
optimiseurDiscrBinaire = torch.optim.Adam(modelDiscr.parameters(), lr=0.001)
optimiseurGen = torch.optim.Adam(modelGen.parameters(), lr=0.001)

iteration = 10000

# On convertie les labels fourni dans la database en tensor
Y_train_10 = Y_train_tensor[:iteration].to(device)
X_train_10 = X_train_tensor[:iteration].to(device)

for i in range(iteration):

    if i + batchsize > len(X_train_tensor):
        break

    discrtotal_loss=0.0

     # Initialisation des gradients
    optimiseurDiscr.zero_grad()
    optimiseurDiscrBinaire.zero_grad()
    
    # Sélection d'un lot d'images réelles
    real_img = X_train_tensor[i * batchsize:(i + 1) * batchsize].to(device)

    #Creation des labels (pour non conditionnel)
    #real_labels = torch.ones(batchsize,1).to(device)
    #fake_labels = torch.zeros(batchsize,1).to(device)

    #Creation des labels (pour conditionnel)
    choicelabels = create_labels(batchsize,3)
    TestErrorlabels = create_labels(batchsize,1)
    real_labels = torch.ones(batchsize,1).to(device)
    fake_labels = torch.zeros(batchsize,1).to(device)
    
    #Entrainement conditionnel
    #passage dans le CNN des vrais images pour conditionnelle
    prediction_real   = modelDiscr.forwardPropagation(real_img,Y_train_10[i].unsqueeze(0),True,False)
    #On calcul la loss function uniquement sur la partie des images réels et on attribue un label reel
    optimiseurDiscr.zero_grad()
    loss_cond = lossFunctionCondit(prediction_real,Y_train_10[i].unsqueeze(0))
    loss_cond.backward()
    optimiseurDiscr.step()
    optimiseurDiscr.zero_grad()
    
    
    #Entrainement Binaire
    #passage dans le CNN des vrais images pour binaire (vrai faux)
    prediction_realBinaire = modelDiscr.forwardPropagation(real_img, create_labels(batchsize,Y_train_10[i]), False, False)
    #On calcul la loss function uniquement sur la partie des images réels et on attribue un label reel
    if Y_train_10[i]==chiffreRecrée:
        loss_Bin_Real = lossFunction(prediction_realBinaire,real_labels)
    else:
        loss_Bin_Real = lossFunction(prediction_realBinaire,fake_labels) 
    #loss_Bin_Real.backward()    
    
    #Generation d'une image fausse
    noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
    fake_img=modelGen(noise,choicelabels)
    # # #passage de l'image fausse dans le discriminateur et on mets le fake label dessus
    prediction_fake = modelDiscr.forwardPropagation(fake_img,choicelabels,False,False)
    loss_Bin_Fake = lossFunction(prediction_fake,fake_labels)

    optimiseurDiscrBinaire.zero_grad()
    discrlossBin = loss_Bin_Real+loss_Bin_Fake #mesure du discriminateur pour l'optimiseur
    discrlossBin.backward()
    optimiseurDiscrBinaire.step()
    optimiseurDiscrBinaire.zero_grad()
    

    # print(f"Iteration {i}, loss: {discrlossBin}")


    #entrainement generateur
     # on entraine le générateur tous les 5 iterations
        
    if i % 5 == 0 :
        optimiseurGen.zero_grad()
        # #Generation d'une image
        noise = torch.randn(batchsize,100,1,1).to(device) # 64 échantillons, 100 dimensions, 1x1 (h, w)
        gen_img=modelGen(noise,choicelabels)
        # # #passage de l'image fausse dans le discriminateur et on mets le fake label dessus

        prediction_gen = modelDiscr.forwardPropagation(gen_img,choicelabels,False,True)
        loss_gen = lossFunction(prediction_gen,real_labels)
        
        loss_gen.backward()
        optimiseurGen.step()
        print(f"Iteration {i}, Loss conditionnelle: {loss_cond.item()}, Loss binaire: {discrlossBin.item()}, Gen loss: {loss_gen.item()}") 
        if i % 1000 == 0:  # affichage des images toutes les 100 itérations
            with torch.no_grad():
                gen_images = gen_img.detach().cpu()
                grid = U.make_grid(gen_images, nrow=5, normalize=True)
                plt.imshow(np.transpose(grid, (1, 2, 0)))
                print(choicelabels[0])
                plt.show()

# #affichage
# fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# axs[0].imshow(real_img[-1][0].cpu().detach().numpy(), cmap='gray')
# axs[0].set_title(f'Image d\'entrée (MNIST) / Iteration : {iteration}')
# axs[0].axis('off')

# axs[1].bar(range(11),prediction_real.cpu().detach().numpy()[0])
# axs[1].set_title('Sortie de la couche entièrement connectée')
# axs[1].set_xlabel('Classification (sortie du FCL)')
# axs[1].set_ylabel('Probabilité')

# plt.tight_layout()
# plt.show()



