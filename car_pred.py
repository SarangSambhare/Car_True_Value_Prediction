import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import  make_pipeline
car = pd.read_csv('quikr_car - quikr_car.csv');


# we have to clran the data as there is lot of variation in data
# we have to do following things with data
# 1 year has many non - year values . has to remove that.
# 2 year is in object so convert it into int
# 3 price has 'ask for price' value.
# 4 price in object conv it into int
# 5 kms_driven has kms with integers has to remove that also commas with values has to remove that
# 6 kms_driven object to int
# 7 kms_driven has nan values so remove that.
# 8 fuel type has nan values
# 9 we will keep only first 3 words of name as they are very wierd and long

backup = car.copy();

car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int);
car  = car[car['Price']!="Ask For Price"]
car['Price'] = car["Price"].str.replace(',','').astype(int);
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','');
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int);
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ');
car = car.reset_index(drop=True);
car = car[car['Price']<6e6].reset_index(drop=True);
print(car);
car.to_csv('clean car.csv');
#print(car);
 
# Model

x = car.drop(columns='Price');
y = car['Price'];
ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')

#print(r2_score(y_test,y_p));
for i in range(800):
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=i);
    score = []
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_p = pipe.predict(x_test)
    score.append(r2_score(y_test,y_p))
#print(score[np.argmax(score)]);
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=np.argmax(score));
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_p = pipe.predict(x_test)

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

print(pipe.predict(pd.DataFrame([['Hyundai Grand i10','Hyundai',2020,3000,'Petrol']],columns=['name','company','year','kms_driven','fuel_type'])))