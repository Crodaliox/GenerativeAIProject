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

# On load le database d'image qui servira d'entrainement
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# X_train= image
# Y_train = etiquette : indique quel chiffre (de 0 à 9) est représenté dans chaque image.

# Normalisation des images de la database entre 0 et 1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

#Methode pour convertir les images par des tensor

X_train_tensor = torch.tensor(X_train)[:, np.newaxis, :, :] #images sous forme de tenseur à 4 dimensions ((batch_size, channels, height, width)
Y_train_tensor = torch.tensor(Y_train).long()  # .long mets chaque valeur a 64 bits
print(X_train_tensor.shape)
print(Y_train_tensor.shape)

#entrainement !
# Sélectionner les 10 premières images pour l'entraînement
# X_train_10 = X_train_tensor[:10]
# Y_train_10 = Y_train_tensor[:10]
