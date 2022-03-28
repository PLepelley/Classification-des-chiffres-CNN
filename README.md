# Classification-des-chiffres-CNN

# CNN_MSINT
Premier modèle CNN avec l'aide du jeu de données MSINT 

Ce readme présente le projet en deux parties. la première est théorique: elle reprend les bases du deep learning, propose une approche schématique du fonctionnement des réseaux neuronaux utilisés en deep learning, et explicite l'architecture CNN. La deuxième est pratique, et est constituée du brief à réaliser avec le jeu de données MNIST. 

Les explications techniques suivantes viennent du livre de François Chollet *Deep Learning with Python*. 


## **1. Réseau de neurones**
*Dans cette partie, nous reviendrons dans les grandes lignes sur le fonctionnement d'un réseau neuronal artificiel, en expliquant en premier lieu le principe du perceptron*

### 1.1 Le perceptron 

Le perceptron, ou neurone artificiel tient son nom de deux éléments importants: la structure interne d'une unité neuronale, et les interconnexions existantes entres-elles. Voici une forme schématique de neurone artificiel:

![image](https://user-images.githubusercontent.com/95342035/160400635-0957019b-5298-472a-b8a4-aa6e3e9e7840.png)

Une perceptron est relié à un nombre *n* d'entrées (ou input, numérotés ici X0, X1 et Xn) déterminés par un poids synaptique (*note: on reprend ici la sémantique du du réseau cérébral humain*) *W*. Chaque input doit être multiplié par son poids (ici par exemple, nous aurons X0 multpilié par W0). Un biais (noté b) peut-être additionné à la somme de tous les inputs par leur poids respectifs. 

Le résultat de la somme est "filtré" dans une fonction dite "d'activation", ici notée *Fa*. La fonction d'activation est généralement une fonction sigmoïde, comme pour une régression classique, mais on lui préfère la fonction RELu, représentées graphiquement ci-dessous. 

![image](https://user-images.githubusercontent.com/95342035/160403614-7e975c29-229b-4d51-878d-27579ce11f95.png)

Nous venons d'expliquer le principe de fonctionnement d'un perceptron. Ce dernier, dans le cadre d'une expérience de Deep Learning, est mis en relation avec plusieurs autres perceptrons, pour maximiser l'apprentissage ainsi que le résultat. On parle alors de **Perceptron multi-couches** ou en anglais *Multi-layer Perceptron (MLP)*. En voici une illustration: 

![image](https://user-images.githubusercontent.com/95342035/160404344-66a12bce-3fcd-4f49-9a45-be4e3d685827.png)

Le schéma ci-dessus est constitué de trois niveaux de perceptrons, où trois *couches*. 

1) La couche d'entrée (*input layer*) qui reçoit les vecteurs d'entrée (*input vectors*)

2) La couche cachée (*hidden layer*) sur laquelle nous n'avons pas de prise: c'est la "boite noire".

3) La couche de sortie (*output layer*), qui produit les outputs. 

Chaque neurone est relié à la couche de neurones suivantes. Nous avons donc ici deux matrices de poids, nomément W (*=Wij*) et H(*=Hjk*). Ainsi, en reprenant le calcul effectué avec un seul perceptron et en l'adaptant au MLP, on obtient:  

![image](https://user-images.githubusercontent.com/95342035/160406552-34703f49-4a66-4c9e-bddc-58ae1628a526.png)

Nous voyons qu'avec seulement trois couches, la phase de calcul de l'entrée à la sortie du réseau neuronal s'avère complexe. Dès lors, comment faire pour déterminer les valeurs de tous les poids et biais synaptiques? Dans ce cas de figure, nous utilisons un algorithme dit de **rétropropagation du gradient** (*back-propagation* en anglais). 

Pour se faire, on commence par initialiser les poids du réseau de manière aléatoire. On obtient donc en sortie un résultat *y*. Ensuite, on calcule l'erreur entre la sortie donnée prédite par le réseau et la valeur cible ou *target*:

![image](https://user-images.githubusercontent.com/95342035/160408835-46809765-1d9e-4f57-b9c3-e6fa4a6e430a.png)

Adapté à l'intégralité de notre réseau neuronal, on obtient: 

![image](https://user-images.githubusercontent.com/95342035/160409590-1cf0ec08-31ae-4487-8833-a79dd0744d8a.png). 

De manière générale, on utilise la méthode dite de descente stochastique de gradients, qui utilise des fournées (ou *batches* de valeurs d'entrée) au lieu de prendre en compte l'intégralité du dataset. 


## **2. Réseau de neurones convolutif (CNN)**

Dans le cadre d'une expérience de Deep Learning, plusieurs architectures sont mises à disposition selon l'objectif recherché. Pour la détection d'images, nous nous intéresserons à l'architecture CNN (*convolutional neural network*, ou réseau de neurones convolutif), spécialement conçue pour traiter des images en entrée.  

Un réseau de neurones convolutif est composé de quatre couches: la couche de convolution, la couche de *pooling*, la couche de correction ReLU et la couche *fully-connected*. 

### 2.1 La couche de convolution 
Prenons pour exemple deux images: 

![image](https://user-images.githubusercontent.com/95342035/160417184-0c17a1ae-2ff5-4be9-8e66-81ee237d2970.png)

Ces dernières sont semblables sur certains asects, mais présentent des différences: 

![image](https://user-images.githubusercontent.com/95342035/160417313-7530ebe8-d225-4908-8de6-1e5e281cb711.png)

Ces "morceaux" vont être appelés features. Dans le cadre d'une convolution, on va faire "glisser" chaque feature sur l'image:

![image](https://user-images.githubusercontent.com/95342035/160417756-03c2609e-de25-4d9f-ae94-346c9640a4f4.png)

Chaque features va produire une convolution. Ainsi, plus nous aurons de features, plus nous aurons de convolutions. 

![image](https://user-images.githubusercontent.com/95342035/160418307-f0d78d65-6b63-43ba-a74d-9e4a4ae23c3c.png)


### 2.2 La couche de correction ReLU
La fonction ReLU permet tout simplement de transformer En 0 toutes les valeurs négatives et de conserver les valeurs positives. Pour la représentation graphique de cette fonction, voir plus haut. 

![image](https://user-images.githubusercontent.com/95342035/160418438-2c1abd6e-f35d-453c-afe2-83e566610eef.png)


### 2.3 La couche de pooling

Le pooling peut-être divisé en deux sous-catégories: le max-pooling où le mean-pooling. Le max pooling utilisé ici prend la valeur maximale de chaque "morceau d'image":

![image](https://user-images.githubusercontent.com/95342035/160418833-55ebf231-817e-4404-897c-bab499f7e4fb.png)

Appliqué à nos matrices précédentes, nous obtenons: 

![image](https://user-images.githubusercontent.com/95342035/160418983-f91bf8e4-52a9-40f0-a52c-ec2d6da645e0.png)

### 2.4 Le flattening

Le flattening où mise à plat consiste à prendre la totalité des valeurs de nos matrices précédemment calculées et à les empiler, en vue de les exploiter dans la couche d'entrée d'un réseau de neurones. 

![image](https://user-images.githubusercontent.com/95342035/160419528-92571761-198b-4aa3-b3f1-c28c17e666d5.png)

Une fois mis à plat, nous obtenons un réseau de neurones dit **fully connected**

Nous pouvons représenter de manière schématique l'architecture CNN de la manière suivante: 

![image](https://user-images.githubusercontent.com/95342035/160420306-8f67dc77-20d2-4f09-a5ca-eb6d99ae7ce1.png)

