from statistics import mode
import pandas as pd
import numpy as np
import scipy.stats 
import algoritmoEntrenamiento as ae

def get_modes(list_of_lists):
    result = []
    num_lists = len(list_of_lists)
    list_length = len(list_of_lists[0])  # Assuming all lists have the same length

    for i in range(list_length):
        elements = [lst[i] for lst in list_of_lists]    
        result.append(mode(elements))

    return result

def prediccionAlg(datosTesteo, conjunto):
    l = list()
    for i in conjunto:
        print(i[0].predict(datosTesteo[i[1]]))
        pred = i[0].predict(datosTesteo[i[1]])
        l.append(pred)
    return  get_modes(l)


titanic = titanicTestingData = pd.read_csv('./datos/titanic_test.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])





listaModelosEntrenados = ae.entrenamientoDeModelos(titanic, 3, 'tree',0.66, "titanicPrueba")

print (prediccionAlg(titanic, listaModelosEntrenados))

