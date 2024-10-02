#Projet DeepLearning

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

#Chargement d'une image
image = Image.open("lib\imgTest\lemans.jpg")

transform = transforms.Compose([
    #transforms.Resize((128, 128)),  # Redimensionnement l'image à 128x128 pour simplifier
    transforms.ToTensor()           # Convertir en tenseur PyTorch (matrice de donnée)
])
input_image = transform(image).unsqueeze(0)  

# Définition de la couche de convolution
conv1 = nn.Conv2d(3, 3, 4, 2, 1, bias=False)
output = conv1(input_image)

# Visualisation de l'image d'entrée et les cartes d'activation en sortie
# Convertir le tenseur d'entrée en image pour affichage
fig, axs = plt.subplots(1, 4, figsize=(12, 4))

# Afficher l'image originale
axs[0].imshow(input_image[0].permute(1, 2, 0))  # Permuter les dimensions pour [H, W, C]
axs[0].set_title('Image originale')

# Afficher les 3 cartes d'activation en sortie (une par filtre)
for i in range(3):  # Parce que la sortie a 3 canaux
    axs[i+1].imshow(output[0, i].detach().numpy(), cmap='gray')  # Afficher chaque canal
    axs[i+1].set_title(f'Activation {i+1}')

# Désactiver les axes
for ax in axs:
    ax.axis('off')

plt.show()

# Définition de l'architecture CNN
class CNN(nn.Module):
  def __init__(self):
    super().__init__() #Instantiation des methodes de la classe parent (nn.module)

    # Creation de filtre convolutif avec noyau (explicationCNN/9OZKF.gif) (https://www.youtube.com/watch?v=CXOGvCMLrkA&ab_channel=Anujshah)
        # La methode nn.Conv2d a pour attribut
        # 1 - Le nombre de cannaux par pixels (ici RVB)
        # 2 - Le nombre de cannaux à la sortie (le nombre d'image filtré avec un noyau (kernel) à la sortie)
        # 3 - La taille du noyau (kernel) du filtre appliqué
        # 4 - le stride (le pas) de déplacement du filtre sur toute l'image (par exemple 2 déplace le noyau de 2 pixels vers la droite)
        # 5 - padding (rajoute des pixels autour de l'image pour ne pas la faire retrecir au passage du filtre)
    self.conv1 = nn.Conv2d(  3, 3, 4, 2, 1, bias=False)
    
