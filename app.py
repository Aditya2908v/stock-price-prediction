import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import math
import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock Ticket', 'AAPL')

df =  pd.read_csv(f"C:\\Users\\USER\\Downloads\\Stock-price-predicition-master\\csv files\{user_input}.csv")
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA. & 200')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data=df.filter(['Close'])
dataset=data.values
training_data_len = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model('keras_model.h5')

test_data=scaled_data[training_data_len-60: , : ]
#Create the data sets x_test and y_test

x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

train=data[:training_data_len]
valid=data[training_data_len:]
valid.loc[:, 'predictions'] = predictions


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(16,8))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'] , color='red')
plt.plot(valid['Close'] , color='yellow')
plt.plot(valid[ 'predictions'] , color='green')
plt.legend(['Train','Validation', 'predictions'], loc='lower right')
st.pyplot(fig2)