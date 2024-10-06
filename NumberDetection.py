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


#Initialisation
#On defini une variable permettant de vérifier si un GPU est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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


#entrainement !
iteration = 5000
# On selectionne 10 img et 10 label
X_train_10 = X_train_tensor[:iteration].to(device)
Y_train_10 = Y_train_tensor[:iteration].to(device)

#Creation du CNN avec une class
class CNN(nn.Module):


    def __init__(self):
            #Dans cette methode, on définit l'ensemble des étapes qui constitue le model CNN
            super(CNN, self).__init__()  # On ajoute les fonction du constructeur de nn.Module
            self.conv1 = nn.Conv2d(1, 64, 5, 1, 0, bias=False)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fcl1 = nn.Linear(64 * 4 * 4, 128)  
            self.fcl2 = nn.Linear(128, 10)
    
        
    def forwardPropagation(self,x):
        #On vient appliquer des fonctions d'activation sur chaque perceptron du model
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpooling1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpooling2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.fcl1(x))
        x = F.softmax(self.fcl2(x),dim=1)

        return x
    


#Boucle d'entrainement
#C'est la ou le model CNN va être entrainé avec des données

#Creation du model :
model = CNN().to(device)

#Creation de la loss function ou la fonction cout (où l'on va calculer l'errer)
    #Comparaison entre les etiquettes du Y_test initial à ceux sortie.

    #Pour l'instant fonction utilisé par pytorch.
    #Elle effectue automatiquement la fonction d'activation de softmax pour chaque valeur
lossFunction = nn.CrossEntropyLoss()  # cette fonction pytorch est adapté à l'emploi multiclass (ce n'est pas exactement la même lossfonction que sur la vidéo)

#Creation de la fonction de descente de gradient (pour améliorer à chaque itération la loss function)
    #Principe : bouger la frontiere de decision pour minimiser la LossFunction
    #Pour ça il faut définir comment réagi la loss fonction en fonction des variations de la frontiere
    #A revoir pour fonctionnement mais on va utiliser un optimizer déjà integré par pytorch pour l'instant
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimiseur Adam (voir tuto) avec un learning rate de 0.001



batchSize = 64 # Les données sont séparé par lot de 64
# Créer un DataLoader pour charger les données par lots et les mélanger pour ne pas avoir toujours la même chose
#train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batchSize, shuffle=True)

for i in range(iteration):

    total_loss=0.0

    lossEvaluation=0.0
    #allBatch = len(train_loader)  # Nombre total de batchs

    # Initialisation des gradients
    optimizer.zero_grad() 

    # On mets les données dans le modèle CNN (il fait toutes les étapes du CNN)
    CNNoutput = model.forwardPropagation(X_train_10[i].unsqueeze(0)) #C'est les 10 proba obtenu après softmax

    #On calcul le niveau d'erreur à la sortie avec la loss Function
    lossValue = lossFunction(CNNoutput,Y_train_10[i].unsqueeze(0)) #comparaison de la sortie avec les labels de Y_train du début

    #Backward Propagation : On mesure comment cette fonction cout varie à chaque couche de notre model en remontant chaque couche
    lossValue.backward()

    #On corrige chaque parametre du model (biais etc...) pour modifier la frontiere de descision grace à la descente de gradient
    optimizer.step() #Il fait la descente de gradient pour ce step

    #(Calcul de la moyenne d'erreur pour chaque batch afin d'avoir la valeur moyenne de perte pour chaque iteration)
    
    total_loss += lossValue.item()

    
    average_loss = total_loss / (i + 1)  # Moyenne jusqu'à l'itération courante
    print(f"Iteration {i + 1}/{iteration}, Erreur moyenne : {average_loss}")

    predicted_class = torch.max(CNNoutput, 1)
    print("Classe prédicte : ", predicted_class.indices.cpu().numpy())


fig, axs = plt.subplots(2, 1, figsize=(8, 12))

axs[0].imshow(X_train_10[-1][0].cpu().detach().numpy(), cmap='gray')
axs[0].set_title('Image d\'entrée (MNIST)')
axs[0].axis('off')

axs[1].bar(range(10), CNNoutput.cpu().detach().numpy()[0])
axs[1].set_title('Sortie de la couche entièrement connectée')
axs[1].set_xlabel('Classification (sortie du FCL)')
axs[1].set_ylabel('Probabilité')

plt.tight_layout()
plt.show()

    
