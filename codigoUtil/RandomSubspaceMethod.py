import pandas as pd
import numpy as np
import scipy.stats 


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])


def RandomSubspaceData (fileName, data, amountNewFiles):
    col = data.loc[:, data.columns != 'Survived']  # cogemos todas las columnas salvo la ultima (que es la label)

    numCol=col.shape[1]

    for i in range (amountNewFiles):
        #cogemos una serie de columnas aleatorias    
        selcted_column=np.random.choice(numCol,size=int(np.sqrt(numCol)),replace=False)
        subspace_sample = col.iloc[:, selcted_column].copy()
        
        route = f'./datos/conjutosEntrenamiento2/{fileName}_trainSet_{i+1}.csv'
        subspace_sample.to_csv(route, index = False)


RandomSubspaceData('titanic', titanicData, 3)