# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY


Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![download](https://user-images.githubusercontent.com/94296805/225692901-0f44c616-0d7b-41e8-8b03-30acbd36780e.png)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM


### Developed By:Javith farkhan S

### Register Number:212221240017
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Digital_image').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head(14)

X = df[['input']].values
y = df[['output']].values
Scaler = MinMaxScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
model = Sequential([
Dense(17,activation = 'relu',input_shape=[1]),
Dense(10,activation = 'relu'),
Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2000)
model.fit(X_train,y_train,epochs=2000)
loss_plot = pd.DataFrame(model.history.history)
loss_plot.plot()
err = rmse()
preds = model.predict(X_test)
err(y_test,preds)
x_n1 = [[40]]
x_n_n = Scaler.transform(x_n1)
model.predict(x_n_n)

```

## Dataset Information

![Dataset_1](https://user-images.githubusercontent.com/94296805/225693193-e4f5374f-752a-4d3d-b293-26183d2741cc.png)



## OUTPUT

### Training Loss Vs Iteration Plot

![LossvsIteration](https://user-images.githubusercontent.com/94296805/225693264-d98cc059-c876-4cb1-80cc-2be0594bd64f.png)



### Test Data Root Mean Squared Error

![RootMeanSquareError](https://user-images.githubusercontent.com/94296805/225693320-6392f858-9c50-4bc6-900c-09d3f980d827.png)



### New Sample Data Prediction


![PredictionData](https://user-images.githubusercontent.com/94296805/225693360-7b8dd3fe-df8c-41ac-a224-ba9b7d0f7a3e.png)


## RESULT

successfully created and trained a neural network regression model for the given dataset
