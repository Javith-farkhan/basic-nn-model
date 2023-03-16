# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![download](https://user-images.githubusercontent.com/94296805/225385723-3c0ee7d8-c017-469f-b448-98b01e8e4f81.png)


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

``

Developed By:Javith farkhan S

Register Number:212221240017

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

![Dataset_1](https://user-images.githubusercontent.com/94296805/225378543-e72689c3-70ec-48c3-8e2d-30d3a766fe20.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![Loss vs Iteration](https://user-images.githubusercontent.com/94296805/225378759-797a81ac-22be-4019-869f-88bfdd532a64.png)


### Test Data Root Mean Squared Error

![Root Mean Square Error](https://user-images.githubusercontent.com/94296805/225378848-69d6b40a-19cb-493f-95ee-b263805737bc.png)


### New Sample Data Prediction

![Prediction Data](https://user-images.githubusercontent.com/94296805/225378969-639c13b2-6d89-4bb3-811c-cc2fb30a5302.png)


## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully
