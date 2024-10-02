#Projet DeepLearning

import numpy as np
import torch
import torch.nn as nn

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
    
