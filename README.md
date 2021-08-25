# Interpretable Corporate Bond Prediction Service using LIME

This project is for DB finance project that predict the debenture availability.
When companies needs money, they want to issue the corporate bond.
However, it is very expensive to test the corporated bond, to solve this problem we propose this service
This service is belong to DongBu finance.

## System
This system predict the credit rating and use them as the corporate bond availability.

There are 3 process:

1. Use unlabeled dataset
``` Python
clf = RandomForestClassifier(max_depth=3, random_state=0).fit(newXtrain,y_train)
for i in range(len(newxunlabeld)):
    if np.max(clf.predict_proba([newxunlabeld[i]]))>0.6:
    listxtrain.append(xunlabeled[i].astype(np.float64))       
    listytrain.extend(clf.predict([newxunlabeld[i]]))
```

2. Feature selection:
Use recursive feature elimination


3. Prediction:
Use random forest
``` Python
from sklearn.ensemble import RandomForestClassifier

# learn Random Forest by using train data which has 5 depth tree
clf1 = RandomForestClassifier(max_depth=5, random_state=743)
clf1.fit(newXtrain,np.array(listytrain))

# get predict value
pred=clf1.predict(realtest)
```


## Serive page
We create service page using node.js and MongoDB.


### Pages
This page is for getting information.

![image](https://user-images.githubusercontent.com/42733881/130744798-784b2cc4-5e2c-4df0-a9d2-14f5e08ee469.png)

This is the explanation by LIME.

![image](https://user-images.githubusercontent.com/42733881/130745389-b03bc104-2010-4164-b0e1-29ecdf2a14e5.png)

![image](https://user-images.githubusercontent.com/42733881/130745443-1fd6d4b7-58a8-42a2-8f75-08316a77b7a4.png)


