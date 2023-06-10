import pandas as pd
import numpy as np
import scipy.stats 


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])

pcosData = pd.read_csv('./datos/pcos.csv', skiprows = 1, header = None,
                           names=['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)',
                                   'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marriage Status (Yrs)',
                                     'Pregnant(Y/N)', 'No. of abortions', 'I beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)',
                                       'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
                                         'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
                                           'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)', 'PCOS (Y/N)'])



def RandomSubspaceData (fileName, data, amountNewFiles):

    col = data[data.columns[:-1]]# cogemos todas las columnas salvo la ultima 
    
    survived=data.loc[:, data.columns == data.columns[-1]] #guardamos la ultima columna

    numCol=col.shape[1] #dimension de la columna en col
    

    for i in range (amountNewFiles):
        #cogemos una serie de columnas aleatorias    
        selcted_column=np.random.choice(numCol,size=int(np.sqrt(numCol)),replace=False)#seleccionamos un subconjunto de columnas 
        subspace_sample = col.iloc[:, selcted_column].copy()#creamos un nuevo DataSet 
        
        #-----------------------------A LO MEJOR HAY QUE CAMBIARLO---------------
        subspace_sample[data.columns.values[-1]]=survived #añadimos columna survived
        #------------------------------------------------------------------------

        route = f'./datos/conjutosEntrenamiento2/{fileName}_trainSet_{i+1}.csv'
        subspace_sample.to_csv(route, index = False)


RandomSubspaceData('pcos', pcosData, 3)
RandomSubspaceData('titanic', titanicData, 3)

