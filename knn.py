import numpy as np
from collections import Counter

import numpy as np
from collections import Counter

# For testing purposes.
from sklearn import datasets
from sklearn.model_selection import train_test_split

# For plotting purposes.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def distancia_euclideana(x1, x2): # Recibe dos arrays 
  return np.sqrt(np.sum((x1-x2)**2))  # Devuelve la suma de las distancias entre los puntos de cada array

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predecir(self, X):
        predicciones = [self._predecir(x) for x in X]
        return predicciones

    def _predecir(self, x):
        # Procesar la distancia de cada elemento al arreglod de entrenamiento
        distancias = [distancia_euclideana(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distancias)[:self.k] # Argsort se usa para un ordenado solamente de los indices en el array.
        k_mas_cercanos_etiquetas = [self.y_train[i] for i in k_indices]

        # Mayoria de etiquetas comunes cercanas
        # Counter crea pares donde se indica la cantidad de veces que se repite un elemento,
        # en este caso se utiliza para saber que etiqueta es la mas comun en la colection de etiquetas de los elementos mas cercanos.
        mas_comun = Counter(k_mas_cercanos_etiquetas).most_common()

        return mas_comun[0][0]
         

# Demostracion
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(data[:,2],data[:,3], c=target, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predicciones = clf.predecir(X_test)

print(predicciones)

acc = np.sum(predicciones == y_test) / len(y_test)
print(acc)