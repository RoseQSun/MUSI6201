import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

# load tempo csv and make 3 categories
dfslow = pd.read_csv('sanyan_tempo.csv')
dfyuanban = pd.read_csv('yuanban_tempo.csv')
dferliu = pd.read_csv('erliu_tempo.csv')
dfmid = pd.concat([dfyuanban,dferliu])
dfliushui = pd.read_csv('liushui_tempo.csv')
dfkuaiban = pd.read_csv('kuaiban_tempo.csv')
dffast = pd.concat([dfliushui,dfkuaiban])
frames = [dfslow, dfmid, dffast]
df = pd.concat(frames)

# training and testing
categories = df[['Speed']]
predictors = df[['Tempo']]
predictors =(predictors-predictors.mean())/predictors.std()
pred_train, pred_test, cat_train, cat_test = train_test_split(predictors, categories, test_size=.33, random_state=5)
multimodel = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 5000)
multimodel.fit(pred_train, cat_train.Speed) 
model_pred = multimodel.predict(pred_test)

# confusion matrix
cm = confusion_matrix(cat_test, model_pred)
cmdf = pd.DataFrame(cm, columns=['Slow','Mid','Fast'], index=['Predicts_Slow','Predicts_Mid','Predicts_Fast'])
# print(cmdf)

#classification report
cr = classification_report(cat_test, model_pred)
# print(cr)

# load tempo csv and make 2 categories
dfslow2 = pd.read_csv('manban_tempo.csv')
dfsm = pd.concat([dfslow2, dfmid])
smcategories = dfsm[['Speed']]
smpredictors = dfsm[['Tempo']]
smpredictors =(smpredictors-smpredictors.mean())/smpredictors.std()
smpred_train, smpred_test, smcat_train, smcat_test = train_test_split(smpredictors, smcategories, test_size=.33, random_state=5)
smmultimodel = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 5000)
smmultimodel.fit(smpred_train, smcat_train.Speed) 
smmodel_pred = smmultimodel.predict(smpred_test)

dfsf = pd.concat([dfslow2, dffast])
sfcategories = dfsf[['Speed']]
sfpredictors = dfsf[['Tempo']]
sfpredictors =(sfpredictors-sfpredictors.mean())/sfpredictors.std()
sfpred_train, sfpred_test, sfcat_train, sfcat_test = train_test_split(sfpredictors, sfcategories, test_size=.33, random_state=5)
sfmultimodel = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 5000)
sfmultimodel.fit(sfpred_train, sfcat_train.Speed) 
sfmodel_pred = sfmultimodel.predict(sfpred_test)

# confusion matrices
smcm = confusion_matrix(smcat_test, smmodel_pred)
smcmdf = pd.DataFrame(smcm, columns=['Moderate', 'Slow'], index=['Predicts_Moderate','Predicts_Slow',])

sfcm = confusion_matrix(sfcat_test, sfmodel_pred)
sfcmdf = pd.DataFrame(sfcm, columns=['Fast', 'Slow'], index=['Predicts_Fast','Predicts_Slow',])

smcr = classification_report(smcat_test, smmodel_pred)
sfcr = classification_report(sfcat_test, sfmodel_pred)

