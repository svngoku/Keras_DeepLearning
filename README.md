# Training to DeepLearning with Keras

Ce repo est un support d'entrainement aux différentes techniques de Deep Learning avec la bibliothèque Keras.

L'ensemble des morceaux de codes proviennent principalement du livre "L'apprentissage du Deep Learning avec Python" de François Chollet , créateur de la bibliothèque Keras et un cours sur Udemy sur les réseaux de neurones avec tensorflow.

## A qui s'adresse ce repo

Ce repo est destiné aux ingénieur de données spécialisé en Deep learning notament avec la librairie Keras.

## TensorFlow 2

TensorFlow est depuis longtemps la bibliothèque open source Python d'apprentissage automatique (ML) la plus populaire. Elle a été développée par l'équipe Google Brain en tant qu'outil interne, mais en 2015, elle a été publiée sous une licence Apache. Depuis lors, elle a évolué en un écosystème plein d'atouts importants pour le développement et le déploiement de modèles. Aujourd'hui, il prend en charge une grande variété d'API et de modules spécialement conçus pour traiter des tâches telles que l'ingestion et la transformation de données, l'ingénierie des fonctionnalités, la construction et le service de modèles, ainsi que bien d'autres.

TensorFlow est devenu de plus en plus complexe. L'objectif de ce livre est de contribuer à simplifier les tâches courantes qu'un data scientist ou un ingénieur ML devra effectuer au cours d'un processus de développement de modèle de bout en bout. Ce livre ne se concentre pas sur la science des données et les algorithmes ; les exemples présentés ici utilisent des modèles pré-construits comme véhicule pour enseigner les concepts pertinents.

## Ecosystème Tensorflow

![img](https://1.bp.blogspot.com/-4C_bx62kOI4/XfE3XQT422I/AAAAAAAABmY/AbMfOO8yzjctmg30IcgOBaU5UmcZNpAtwCLcBGAsYHQ/s1600/model.png)

## L'apprentissage par transfert avec Keras

L'apprentissage par transfert consiste à prendre des caractéristiques apprises sur un problème et à les exploiter sur un nouveau problème similaire. 

Par exemple, les caractéristiques d'un modèle qui a appris à identifier les ratons laveurs peuvent être utiles pour lancer un modèle destiné à identifier les **tanukis**.

L'apprentissage par transfert est généralement utilisé pour les tâches pour lesquelles votre ensemble de données est trop petit pour entraîner un modèle complet à partir de zéro.

L'incarnation la plus courante de l'apprentissage par transfert dans le contexte de l'apprentissage profond est le flux de travail suivant :

- Prenez les couches d'un modèle précédemment formé.
Les geler, afin d'éviter de détruire les informations qu'elles contiennent lors des prochains cycles de formation.
- Ajoutez des couches nouvelles et entraînables par-dessus les couches gelées. Elles apprendront à transformer les anciennes caractéristiques en prédictions sur un nouvel ensemble de données.
- Entraînez les nouvelles couches sur votre jeu de données.
Une dernière étape, facultative, est le réglage fin, qui consiste à dégeler l'ensemble du modèle que vous avez obtenu ci-dessus (ou une partie de celui-ci), et à le réentraîner sur les nouvelles données avec un taux d'apprentissage très faible. Cela peut potentiellement permettre d'obtenir des améliorations significatives, en adaptant de manière incrémentielle les caractéristiques pré-entraînées aux nouvelles données.

## Gérer les couches: comprendre l’attribut entrainable

Les couches et les modèles ont trois attributs de poids :

- **weights** est la liste de toutes les variables de poids de la couche.
- **trainable_weights** est la liste de ceux qui sont censés être mis à jour (via la descente de gradient) pour minimiser la perte pendant la formation.
- **non_trainable_weights** est la liste de celles qui ne sont pas destinées à être entraînées. Typiquement, ils sont mis à jour par le modèle pendant le passage en avant.

Exemple : la couche dense a 2 poids entraînables (noyau et biais).

```python
layer = keras.layers.Dense(3)
layer.build((None, 4))  # Create the weights

print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))
```

```
weights: 2
trainable_weights: 2
non_trainable_weights: 0
```

## Le flux de travail typique de l’apprentissage par transfert

Afin de comprendre le flux de travail typique d'apprentissage par transfert avec Keras, il est important d’énuméré ses étapes:

1. Instanciez un modèle de base et chargez-y des poids pré-entraînés.
2. Geler toutes les couches du modèle de base en définissant `trainable = False`.
3. Créez un nouveau modèle à partir de la sortie d'une (ou plusieurs) couche(s) du modèle de base.
4. Entraînez votre nouveau modèle sur votre nouveau dataset.

Il est important de noter qu'un flux de travail alternatif, plus léger, pourrait également être le suivant :

1. Instanciez un modèle de base et chargez-y des poids pré-entraînés.
2. Exécutez votre nouvel ensemble de données et enregistrez la sortie d'une (ou plusieurs) couches du modèle de base. C'est ce qu'on appelle l'**extraction de caractéristiques**.
3. Utilisez cette sortie comme données d'entrée pour un nouveau modèle, plus petit.

L'un des principaux avantages de ce deuxième flux de travail est que vous n'exécutez le modèle de base qu'une seule fois sur vos données, plutôt qu'une fois par époque d'apprentissage. C'est donc beaucoup plus rapide et moins cher.

Cependant, ce deuxième flux de travail ne vous permet pas de modifier dynamiquement les données d'entrée de votre nouveau modèle pendant la formation, ce qui est nécessaire lors de l'augmentation des données, par exemple. 

L'apprentissage par transfert est généralement utilisé pour les tâches pour lesquelles votre nouveau dataset ne contient pas assez de données pour former un modèle complet à partir de zéro, et dans de tels scénarios, l'augmentation des données est très importante. 

Afin de mieux le comprendre, nous allons donc nous concentrer sur le premier flux de travail.

## Effectuez une série de réglages fins de l'ensemble du modèle.

### Processus de fine-tuning

L'ajustement fin, quant à lui, nécessite non seulement de mettre à jour l'architecture du CNN, mais aussi de le réentraîner pour apprendre de nouvelles classes d'objets.

L'ajustement fin est un processus en plusieurs étapes :

1. Supprimez les nœuds entièrement connectés à la fin du réseau (c'est-à-dire là où les prédictions d'étiquettes de classe sont effectuées).
2. Remplacer les nœuds entièrement connectés par des nœuds fraîchement initialisés.
3. Geler les couches CONV plus tôt dans le réseau (en s'assurant que toutes les caractéristiques robustes précédentes apprises par le CNN ne sont pas détruites).
4. Commencez la formation, mais ne formez que les têtes de couches FC.
5. Optionnellement, dégeler une partie ou la totalité des couches CONV du réseau et effectuer un deuxième passage de formation.

Enfin, dégelons le modèle de base et entraînons le modèle entier de bout en bout avec un faible taux d'apprentissage.

Il est important de noter que, bien que le modèle de base devienne entraînable, il fonctionne toujours en mode inférence puisque nous avons passé `training=False` lorsque nous l'avons appelé lors de la construction du modèle. 

Cela signifie que les couches de normalisation de lot à l'intérieur ne mettront pas à jour leurs statistiques de lot. Si elles le faisaient, elles causeraient des ravages sur les représentations apprises par le modèle jusqu'à présent.

# Early Stopping

L'arrêt précoce est une technique d'optimisation utilisée pour **réduire** **l'overfitting** sans compromettre la précision du modèle. 

L'idée principale de l'arrêt précoce est d'arrêter la formation avant qu'un modèle ne commence à s'adapter de manière excessive. ( *trop généraliser ???*)

# Adam

Adam combine les meilleures propriétés des algorithmes **AdaGrad** et **RMSProp** pour fournir un algorithme d'optimisation qui peut gérer les gradients épars sur des problèmes bruyants. 

Adam est relativement facile à configurer, les paramètres de configuration par défaut donnant de bons résultats pour la plupart des problèmes.

# ModelCheckpoint

Le callback **ModelCheckpoint** est utilisé en conjonction avec l'entraînement utilisant `model.fit()` pour sauvegarder un modèle ou des poids (dans un fichier de contrôle) à un certain intervalle, de sorte que le modèle ou les poids puissent être chargés plus tard pour continuer l'entraînement à partir de l'état sauvegardé.

# Data Augmentation

L'augmentation des données est une méthode qui consiste à créer artificiellement un nouveau jeu de données pour la formation à partir du jeu de données de formation existant afin d'améliorer les performances des réseaux neuronaux d'apprentissage profond avec la quantité de données disponibles. 

C'est une forme de régularisation qui fait que notre modèle généralise mieux qu'avant.
Ici, nous avons utilisé un objet Keras **ImageDataGenerator** pour appliquer l'augmentation des données afin de **traduire**, **redimensionner**, **faire** **pivoter**, etc. les images de manière aléatoire. 

Chaque nouveau lot de nos données est ajusté de manière aléatoire en fonction des paramètres fournis à ImageDataGenerator.

# Finetuning

Le **finetuning** consiste à prendre les poids d'un réseau neuronal formé et à les utiliser comme initialisation pour un nouveau modèle formé sur des données du même domaine (souvent des images, par exemple). Il est utilisé pour :

- accélérer la formation
- surmonter la petite taille de l'ensemble de données

# VGG16 vs Inception V3

Le **VGG16** est sans aucun doute une bonne architecture de réseau neuronal mais il peut ne pas être performant pour les tâches difficiles car il s'agit d'une simple pile de couches convolutionnelles et de couches de max-pooling suivies les unes des autres et enfin de couches entièrement connectées. En d'autres termes, il n'est pas en mesure d'extraire des caractéristiques très complexes. 

Les réseaux d'**Inception** possèdent des modules d'inception qui consistent en des filtres 1X1, également connus sous le nom de convolutions ponctuelles, suivis de couches convolutionnelles avec différentes tailles de filtres appliquées simultanément. Cela permet aux réseaux **Inception** **d'apprendre des caractéristiques plus complexes.** Ils possèdent plus de couches cachées que le VGG16. Par conséquent, ils sont utilisés pour des problèmes plus complexes.

Conclusion (selon moi) : commencez par le VGG16, qui fonctionne la plupart du temps. Mais si vous sentez qu'il ne fonctionne pas bien, essayez l'Inception, qui devrait être plus performant.