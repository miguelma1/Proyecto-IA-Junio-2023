from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import generacionConjuntosEntrenamiento as gener
import pandas as pd
import numpy as np


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])

############################### SEPARAR DATOS DE SU RESULTADO

titanicTestingData = pd.read_csv('./datos/titanic_test.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])


def separarVariables(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def entrenamientoDeModelos(data, numModelos, algoritmo, proporcionColumnas, fileName):
    res = []
# si no esta entre 0 y 1 paro
    if not 0 <= proporcionColumnas <= 1:
        print("El parametro proporcionColumnas debe ser un numero entre 0 y 1")

    else:
    #si el argumento no es un string
        if not isinstance(algoritmo, str): 
            print("El argumento algoritmo no es un String")
    #si es un string
        else:
            #dado el dataset original obtengo una lista de datos de entrenamiento (hay que ver el numModelos)
            training_data = gener.generadorConjutosEntrenamiento(fileName, data, numModelos)
#por cada dato
            for i in training_data:
                #vego que algoritmo se solicita y creo una nueva instancia del mismo
                if algoritmo.upper() == 'TREE':
                    alg = DecisionTreeClassifier()
        
                elif  algoritmo.upper() == 'SGD':
                    alg = SGDClassifier()
    #separo las variables de datos de las objetivos y ajusto entreno el algoritmo
                x, y = separarVariables(i)
                print("el shape original es: ", x.shape)
                num_columns = int(proporcionColumnas * x.shape[1])
                selected_columns = np.random.choice(x.columns, size=num_columns, replace=False)
               
                selectedX = x[selected_columns]
                print("el shape SELECTED es: ", selectedX.shape, selectedX)


                alg.fit(selectedX, y)
#meto en una lista el modelo entrenado y las columnas empleadas en su entrenamiento
                res.append((alg,x.columns))
    return res

#entrenamientoDeModelos(titanicData, 2, 'tree',1, "titanicPrueba")
print("aqui")
print(entrenamientoDeModelos(titanicData, 1, 'tree',0.67, "titanicPrueba"))

#d = entrenamientoDeModelos(titanicData, 2, 'tree',1, "titanicPrueba")

#print(d[0][1])
#print(titanicTestingData[d[0][1]])
#print(d[0][0].predict(titanicTestingData[d[0][1]]))
