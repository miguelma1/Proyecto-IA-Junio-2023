import pandas as pd
import numpy
import estandarizadoFicheros
from sklearn import model_selection

print(estandarizadoFicheros.pcosData.head(5))

titanic_train, titanic_test = model_selection.train_test_split(estandarizadoFicheros.titanicData, test_size = 0.3, random_state = 99)

print("Datos totales", estandarizadoFicheros.titanicData.shape)
print("Datos entrenamiento", titanic_train.shape)
print("Datos test", titanic_test.shape)

print(titanic_test.head(5) , "Primeros 5 del test")

titanic_train.to_csv('./datos/titanic_train.csv', index = False)
titanic_test.to_csv('./datos/titanic_test.csv', index = False)