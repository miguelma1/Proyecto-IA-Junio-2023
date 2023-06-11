import pandas as pd
import numpy as np
from scipy.stats import bootstrap


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])


def bootstrapData (fileName, data, amountNewFiles):
    for i in range (amountNewFiles):
                    #frac = 1 bootstrappea todo el dataSet, si quisiese el 50% haria frac = 0.5
                    #replace = true puede repetir filas

        bootstrap_sample = data.sample(frac = 0.5, replace = True)
        route = f'./datos/conjutosEntrenamiento/{fileName}_trainSet_{i+1}.csv'
        bootstrap_sample.to_csv(route, index = False)



bootstrapData('titanic', titanicData, 3)