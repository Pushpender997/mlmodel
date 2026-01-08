#Model_1:--
import pandas as pd 
Light_sales={
    "price":[120,150,110,125,160,90,115,105,85,180,110,175],
    "units_sold":[320,280,190,360,420,140,300,220,130,390,210,340]
    
    }

df=pd.DataFrame(Light_sales)
x=df[["price"]]
y=df["units_sold"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

prediction=model.predict(x_test)
print(prediction)

print(y_test.values)

from sklearn.metrics import mean_absolute_error
print("MAE :",mean_absolute_error(y_test,prediction))

from sklearn.metrics import mean_squared_error
import numpy as np

mse=mean_squared_error(y_test,prediction)
rmse=np.sqrt(mse)

print("MSE :",mse)
print("RMSE:",rmse)

from sklearn.metrics import r2_score
r2=r2_score(y_test,prediction)
print("R2 :",r2)


#Model_2:--
import pandas as pd
Light_sales={
    "price":[120,150,110,125,160,90,115,105,85,180,110,175],
    "watage":[9,12,18,9,12,60,9,18,60,15,18,15],
    "units_sold":[320,280,190,360,420,140,300,220,130,390,210,340]
    
    }

df=pd.DataFrame(Light_sales)
x=df[["price","watage"]]
y=df["units_sold"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_poly,y_train)

prediction=model.predict(x_test_poly)
print(prediction)

print(y_test.values)

from sklearn.metrics import mean_absolute_error
print("Mae :",mean_absolute_error(y_test,prediction))


from sklearn.metrics import r2_score
r2=r2_score(y_test,prediction)
print("R2 :",r2)

#_model_3
import pandas as pd


data={
"price":[120,150,110,125,160,90,115,105,85,180,110,175],
"unit":[320,280,190,360,420,140,300,220,130,390,210,340],
"revenue":[38400,42000,20900,45000,67200,12600,
            34500,23100,11050,70200,23100,59500]
            
}



df=pd.DataFrame(data)
x=df[["price","unit"]]
y=df["revenue"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

prediction=model.predict(x_test)
print(prediction)

print(y_test.values)

from sklearn.metrics import mean_absolute_error
print("Mae :",mean_absolute_error(y_test,prediction))


from sklearn.metrics import r2_score
r2=r2_score(y_test,prediction)
print("R2 :",r2)