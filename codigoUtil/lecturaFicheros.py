import pandas
import numpy

pcosData = pandas.read_csv('./datos/pcos.csv', skiprows = 1, header = None, #la ruta puede variar
                           names=['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)',
                                   'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marriage Status (Yrs)',
                                     'Pregnant(Y/N)', 'No. of abortions', 'I beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)',
                                       'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
                                         'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
                                           'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)', 'PCOS (Y/N)'])


titanicData = pandas.read_csv('./datos/titanic.csv', skiprows = 1, header = None, #la ruta puede variar
                              names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Initial', 'Age_band', 'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married', 'Survived'])


print(pcosData.shape)
print(pcosData.head(5))

print(titanicData.shape)
print(titanicData.head(5))