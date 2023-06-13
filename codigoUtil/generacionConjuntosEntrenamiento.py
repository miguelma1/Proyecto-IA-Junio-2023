import pandas as pd
import numpy as np
from scipy.stats import bootstrap


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])




def generadorConjutosEntrenamiento(fileName, data, amountFiles):

    res= [] #lista de conjuntos de entrenamientos

    col = data[data.columns[:-1]]# cogemos todas las columnas salvo la ultima 
    
    objectiveVariable = data.loc[:, data.columns == data.columns[-1]] #guardamos la ultima columna

    numCol=col.shape[1] #dimension de la columna en col

    #hacemos bootstrapping
    for i in range (amountFiles):
        bootstrap_sample = data.sample(frac = 0.5, replace = True) 
        #para cada subset consideramos solo la mitad de los datos de entrada y con posibilidad de repeticion
        for j in range (amountFiles): #para cada nuevo archivo bootstrappeado
            #cogemos una serie de columnas aleatorias    
            selcted_column=np.random.choice(numCol,size=int(np.sqrt(numCol)),replace=False)#seleccionamos un subconjunto de columnas 
            subspace_sample = col.iloc[:, selcted_column].copy()#creamos un nuevo DataSet 
            #-----------------------------A LO MEJOR HAY QUE CAMBIARLO---------------
            subspace_sample[data.columns.values[-1]] = objectiveVariable #añadimos columna survived
            #------------------------------------------------------------------------

            res.append(subspace_sample)

            route = f'./datos/conjuntosEntrenamiento/{fileName}_trainSet_{i+1}.{j+1}.csv'
            subspace_sample.to_csv(route, index = False)
            
    print(res)
    return res





generadorConjutosEntrenamiento("titanic", titanicData, 2)