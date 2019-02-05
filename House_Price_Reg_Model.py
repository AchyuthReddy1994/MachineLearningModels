#importing lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as ny

#importing dataset
data=pd.read_csv("Data.csv")
y=pd.DataFrame(data.iloc[:,2].values)
X=pd.DataFrame(data.iloc[:,3 :].values)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#model building
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

#predictions
y_pred=reg.predict(X_test)


#Backword Elimination
import statsmodels.formula.api as smf
X=ny.append(arr=ny.ones((21613,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17]]
reg_ols=smf.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary()

'''#################################################################################'''

X=X[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#model building
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

#predictions
y_pred_2=reg.predict(X_test)