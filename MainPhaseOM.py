# !python.exe
# coding: cp1251
''' -*- coding: utf-8 -*- '''

from __future__ import with_statement
import os
from datetime import datetime, date, time
from time import *
from types import *
import sip

sip.setapi('QVariant', 2)
from PyQt4 import QtCore, QtGui

from pandas import read_csv, read_excel, DataFrame
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split

def ToPrintLog (sMess):
    print str(datetime.now().strftime("%d.%m.%Y %H:%M:%S ")) + str(sMess)

dataset = read_excel("EnergyEfficiency\ENB2012_data.xls", sheet_name='Data1', index_col=None, na_values=['NA'])
#dataset.head()
ToPrintLog ("Количество наблюдений : " + str(dataset.Y1.count()))
print dataset.head()
# Список имён признаков
ListPRZ = dataset.columns._array_values()

# normalize the data attributes
X  = dataset[ListPRZ]
Y1 = dataset[[u'Y1']]

# X  = dataset[:,0:7]
# Y1 = dataset[:,8]

# Y1 = dataset[[8]].ravel([8])

Y2 = dataset[[u'Y2']]
Y1Y2 = dataset[[u'Y1',u'Y2']]

normalized_X  = preprocessing.normalize(X)
# print normalized_X
normalized_Y1 = preprocessing.normalize(Y1)
normalized_Y2 = preprocessing.normalize(Y2)
# standardize the data attributes
standardized_X = preprocessing.scale(dataset[ListPRZ])

mcorr = dataset.corr()
print "-=:: Матрица корреляций ::=-"
print mcorr
mcorr.to_excel("EnergyEfficiency\ENB2012_corr.xls", sheet_name=u'Корреляции')

model = ExtraTreesClassifier()
# model.fit(normalized_X, Y1)
model.fit(X,Y1Y2)

# Показать степени существенности каждого признака-предиктора

print "-=:: Существенность признаков ::=-"
print(model.feature_importances_)

# Удаление столбцов с минимальными корреляциями с целевыми факторами
modeli         = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, Y1Y2)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)



# dataset = dataset.drop(['X1','X4'], axis=1)
# print dataset.head()

print ("          -= :: END - КОНЕЦ :: =-")