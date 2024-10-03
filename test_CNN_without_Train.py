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

# Load the data and split it between train and test sets
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Normalisation des images de la database entre 0 et 1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# plot a sample
input_imageMnist = X_train[1]
# plot the sample
figMnist = plt.figure
plt.imshow(input_imageMnist, cmap='gray')
# plt.show()

#Chargement d'une image
#image = Image.open("lib\imgTest\lemans.jpg")

#Methode pour convertir les images par des tensor
imgToTensor=transforms.ToTensor()
input_image = imgToTensor(input_imageMnist).unsqueeze(0)  

#1ere itération de la convolution 2D
# Creation de filtre convolutif avec noyau (explicationCNN/9OZKF.gif) (https://www.youtube.com/watch?v=CXOGvCMLrkA&ab_channel=Anujshah)
        # La methode nn.Conv2d a pour attribut
        # 1 - Le nombre de cannaux par pixels (ici RVB)
        # 2 - Le nombre de cannaux à la sortie (le nombre d'image filtré avec un noyau (kernel) à la sortie)
        # 3 - La taille du noyau (kernel) du filtre appliqué
        # 4 - le stride (le pas) de déplacement du filtre sur toute l'image (par exemple 2 déplace le noyau de 2 pixels vers la droite)
        # 5 - padding (rajoute des pixels autour de l'image pour ne pas la faire retrecir au passage du filtre)

conv1 = nn.Conv2d(1, 64, 5, 1, 0, bias=False)
output = conv1(input_image) #Leaky_relu mets toutes les valeurs négatifs à un petit coef (0.01) 
print(output.size())

# Appliquer un pooling maximal sur l'image
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output_pool = maxpool(output)

fig, axs = plt.subplots(1, 65)

for ax in axs:
    ax.axis('off')
    

# axs[0].imshow(input_image[0].permute(1, 2, 0), cmap='gray')  # Permuter les dimensions pour [H, W, C]
# axs[0].set_title('Image originale')

# for i in range(6):  # Parce que la sortie a 3 canaux
#     axs[i+1].imshow(output[0, i].detach().numpy(), cmap='gray')  # Afficher chaque canal
#     axs[i+1].set_title(f'cannal {i+1}')

# # Afficher les cartes d'activation après pooling
# fig2, axs2 = plt.subplots(1, 6, figsize=(12, 4))
# for i in range(3):
#     axs2[i].imshow(output_pool[0, i].detach().numpy(), cmap='gray')
#     axs2[i].set_title(f' Pool {i+1}')

# for ax in axs2:
#    ax.axis('off')


# 2eme itération de convolution
conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
outputconv2 = conv2(output_pool)
print(outputconv2.size())

#Max Pooling de la conv2
output_pool2 = maxpool(outputconv2)

#Affichage
for i in range(64):  # Parce que la sortie a 3 canaux
    axs[i].imshow(output_pool2[0, i].detach().numpy(), cmap='gray')  # Afficher chaque canal
    # axs[i].set_title(f'End {i+1}')

#On mets le resultat en une seul dimension avant de passer dans la couche fully connected
output_flat = output_pool2.view(-1, 64 * 4 * 4)  # Flatten (mise a 1 dimension) (1x1024) : view permet de manipuler la forme d'un tensor pytorch (-1 permet de)
print("Après flatten:", output_flat.size())

#Creation de la couche "fully connected layer"

# Création de la couche entièrement connectée
fcl = nn.Linear(1024, 128)
outputFCL = F.leaky_relu(fcl(output_flat))

# Sortie (classification avec 10 potentielle valeur)
fclBin = nn.Linear(128, 10)
outputExit = F.softmax(fclBin(outputFCL),dim=1)  # Application de la fonction d'activation softmax en sortie pour pouvoir classer selon des probabilités les différentes sortie.
 # Obtenir la classe prédite
predicted_class = torch.max(outputExit, 1)
print("Classe prédicte : ", predicted_class)
fig, axs = plt.subplots(3, 1, figsize=(8, 12))


axs[0].imshow(input_image[0,0].detach().numpy(), cmap='gray')
axs[0].set_title('Image d\'entrée (MNIST)')
axs[0].axis('off')


axs[1].imshow(output_pool2[0].detach().numpy(), cmap='gray')
axs[1].set_title('Activations après Max Pooling')
axs[1].axis('off')


axs[2].bar(range(10), outputExit.detach().numpy()[0])
axs[2].set_title('Sortie de la couche entièrement connectée')
axs[2].set_xlabel('Neurones')
axs[2].set_ylabel('Activation')

plt.tight_layout()
plt.show()

    