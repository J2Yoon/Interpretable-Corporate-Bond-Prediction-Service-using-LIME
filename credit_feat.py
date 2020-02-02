import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import lime.lime_tabular as lime

def getdata(fname):
    data=pd.read_csv(fname)
    size=len(data)
    y=data['등급'].tolist()
    xcol=data.drop(columns="등급").columns.tolist()
    x=data.drop(columns="등급").values.tolist()
    return x,y,size,xcol

def encoding(xtrain, xtest,ytrain,ytest):
    le=LabelEncoder()
    yle=LabelEncoder()
    xtrain[:,0]=le.fit_transform(xtrain[:,0])
    xtest[:,0] = le.fit_transform(xtest[:,0])
    ytrain= yle.fit_transform(ytrain)
    ytest = yle.fit_transform(ytest)
    return xtrain, xtest,ytrain,ytest

X,y,size,xcol=getdata("onlyCredit.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
xtrain,xtest,y_train,y_test=encoding(np.array(X_train),np.array(X_test),y_train,y_test)

score=list()
estimator = RandomForestClassifier(max_depth=3, random_state=0)
# for i in range(60):
#     selector = RFE(estimator, i+1, step=1)
#     selector = selector.fit(xtrain.astype(np.float64), y_train.astype(np.int))
#
#     newXtrain=selector.transform(xtrain)
#     newxtest=selector.transform(xtest)
#
#     clf = RandomForestClassifier(max_depth=3, random_state=0)
#     clf.fit(newXtrain,y_train)
#
#     score.append(accuracy_score(y_test, clf.predict(newxtest)))
#
# print(np.argmax(score))

selector = RFE(estimator, 37, step=1)
selector = selector.fit(xtrain.astype(np.float64), y_train.astype(np.int))

newXtrain=selector.transform(xtrain)
newxtest=selector.transform(xtest)

clf = RandomForestClassifier(max_depth=3, random_state=0).fit(newXtrain,y_train)


print(selector.get_support(True))
xlist=selector.get_support(True)
pred=clf.predict(newxtest)

print(pred)
print(accuracy_score(y_test, pred))

newxcol=list()
for i in xlist:
    newxcol.append(xcol[i])

print(newxcol)

newimfeat=list()
for i in newxcol:
    imfeat=i.split("/")
    newimfeat.append(imfeat[1])

class_name=["A1","A2","B"]
expainer=lime.LimeTabularExplainer(newXtrain.astype(np.float64),feature_names=newimfeat,class_names=class_name,
                          categorical_names="회사명",kernel_width=3)

exptest=newxtest.astype(np.float64)

for i in range(len(exptest)):
    exp=expainer.explain_instance(exptest[i],clf.predict_proba,num_features=5,labels=[0,1,2])
    print(pred[i])
    if pred[i]!=0:
        exp.save_to_file("lime_result"+str(i)+".html",labels=[int(pred[i]),int(pred[i]-1)])
    else:
        exp.save_to_file("lime_result" + str(i) + ".html", labels=[pred[i]])