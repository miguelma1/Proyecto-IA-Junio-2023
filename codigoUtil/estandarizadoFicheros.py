#las rutas pueden variar si se utiliza este codigo en el notebook

import pandas as pd
import numpy

pcosData = pd.read_csv('./datos/pcos.csv', skiprows = 1, header = None,
                           names=['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)',
                                   'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marriage Status (Yrs)',
                                     'Pregnant(Y/N)', 'No. of abortions', 'I beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)',
                                       'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
                                         'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
                                           'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)', 'PCOS (Y/N)'])


titanicData = pd.read_csv('./datos/titanic.csv', skiprows = 1, header = None,
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])


from sklearn.preprocessing import StandardScaler

#variables booleanas o respuesta
titanic_excluded = ['Sex', 'Alone', 'Is_Married', 'Survived']
pcos_excluded = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)','Reg.Exercise(Y/N)', 'PCOS (Y/N)']

#copio la tabla sin la variables que no requieren de estandarizacion
titanic_copy = titanicData.drop(titanic_excluded, axis = 1)
pcos_copy = pcosData.drop(pcos_excluded, axis = 1)

#cojo las columnas que se van a estandarizar
titanic_numerical_columns = [col for col in titanicData.columns if col not in titanic_excluded]
titanic_to_standarize = titanicData[titanic_numerical_columns]

pcos_numerical_columns = [col for col in pcosData.columns if col not in pcos_excluded]
pcos_to_standarize = pcosData[pcos_numerical_columns]

#estandarizo los datos
scalerTitanic = StandardScaler()
titanic_to_standarize = scalerTitanic.fit_transform(titanic_to_standarize)
titanicData[titanic_numerical_columns] = titanic_to_standarize

scalerPcos = StandardScaler()
pcos_to_standarize = scalerPcos.fit_transform(pcos_to_standarize)
pcosData[pcos_numerical_columns] = pcos_to_standarize

#creo los nuevos csv estandarizados
titanicData.to_csv('./datos/standarizedTitanic.csv', index = False)
pcosData.to_csv('./datos/standarizedPcos.csv', index = False)