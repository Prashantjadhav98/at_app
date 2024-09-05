import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# -------<<< Load the trained LSTM model  >>>--------------------
with open('LSTM_Scaled_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# -------<<< Mean Absolute Percentage Error  >>>--------------------
def MAPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test-y_pred)/y_test))*100 

# -------<<< Function to make predictions >>>--------------------
def predict_future(data, n_future=30):
    
    predictions = []
    input_data = data[-1]
    Output_list=[]
    
    for _ in range(n_future):
        pred = model.predict(np.array([input_data]))
        predictions.append(pred[0, 0])
        # Update the input data with the new prediction
        input_data = np.roll(input_data, -1,axis=0)
        input_data[-1] = pred
        print(scaler.inverse_transform(input_data.reshape(1,-1)))
        input_data = np.roll(input_data, -1)
    print('input_data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',scaler.inverse_transform(input_data.reshape(1,-1)))    
    return predictions
# -------<<< Title For Web Ui >>>--------------------
st.title("Time Series Forecasting")

st.write("""
### Predict Future Price Of Sulphur
""")
# -------<<< Upload the input data >>>--------------------
uploaded_file = st.file_uploader("Sulphur.csv", type="csv")

# -------<<< Data Preprocessing and Scaling >>>--------------------
if uploaded_file is not None:
    df= pd.read_csv(uploaded_file)
    df1=df.copy()
    st.subheader('Data preview:')
    st.write(df)
    df['Sulpur_rate'][df['Sulpur_rate'] == 0] = np.nan
    k = 5  # Number of neighbors to consider
    imputer = KNNImputer(n_neighbors=k)
    df_subset = df[['Sulpur_rate']]
    df_imputed = imputer.fit_transform(df_subset)
    df['Sulpur_rate'] = df_imputed
    df['Sulpur_rate'].isna().sum()
    Adj_price=df[['Sulpur_rate']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(Adj_price)
    st.subheader('Scaled_data preview:')
    st.write(scaled_data)

# ----------<<< Creating Sequence Function >>>--------------------
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# ----------<<< Feature lenght and Data Splitting >>>--------------------
feature_length = 5  #(seq_length) Important Feature Need To pass carefully, should be same as we used during model training.
x, y = create_sequences(scaled_data, feature_length)
# x.shape, y.shape
x = x.reshape(x.shape[0], -1)
y=  y.reshape(y.shape[0], -1)
split = int(len(x) * 0.9)
X_train, X_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# ----------<<< prediction on testing Data >>>--------------------
predict_1= model.predict(X_test)
y_pred=scaler.inverse_transform(predict_1)
y_test_inv = scaler.inverse_transform(y_test)
future_date=len(y_pred)
index=df.index[-(future_date):]

# -------<<< Generate a date range for the index in length of test data >>>--------------------
start_date = st.date_input("Select the start date", value=datetime(2023,8,18))
date_range = pd.date_range(start=start_date, periods=len(y_test_inv), freq='D')
df_pred=pd.DataFrame(index=date_range)

df_pred['y_act']=y_test_inv
df_pred['y_pred']=y_pred
df_pred[['y_act','y_pred']].plot()
st.subheader('Actual Price vs Predicted price')
fig = plt.figure(figsize=(12, 5))
plt.plot(df_pred)
plt.legend(["actual rate of sulphur","Predicted rate of sulphur"])
st.pyplot(fig)

mape_value = MAPE(y_test_inv, y_pred)
st.write("### MAPE: {:.2f}%".format(mape_value))

# ---------------------<<< Slider for Fututr prediction >>>--------------------
n_future = st.slider("Select number of days to predict", 1, 60) 
predictions = predict_future(X_test, n_future=n_future)
predictions_array = np.array(predictions)
predictions_reshaped = predictions_array.reshape(-1, 1)
fut_pred = scaler.inverse_transform(predictions_reshaped)
print(fut_pred)
# ---------------------<<< Generate future dates for prediction >>>--------------------
future_dates = pd.date_range(start=df_pred.index[-1] + timedelta(days=1), periods=n_future, freq='D')
fut_pred = fut_pred.flatten()    

# ---------------------<<< Create DataFrame for future predictions >>>--------------------
df_future = pd.DataFrame({
        'Date': future_dates,
        'Predicted': fut_pred
    })
df_future.set_index('Date', inplace=True)

 # ----------<<< Plot the actual vs predicted prices including future predictions >>>--------------------   
st.subheader('Future Forecasted Price Plot')
st.write(f"Forecast for next {n_future} days:")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_pred.index, df_pred['y_act'], label="Actual Data")
ax.plot(df_pred.index, df_pred['y_pred'], label="Predicted Data")
ax.plot(df_future.index, df_future['Predicted'], label="Future Predictions", color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Sulphur Price')
ax.legend()
st.pyplot(fig)

# ---------------------<<< OVERALL DATA vs PREDICTED PRICE >>>--------------------
# df2=pd.DataFrame(index=df.index)
# df2['y_act']=df[['Sulpur_rate']]
# df2['y_pred']=y_pred
# st.subheader('Actual Price vs Predicted price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(df2)
# plt.legend(["actual rate of sulphur","Predicted rate of sulphur"])
# st.pyplot(fig)


# ----------<<< BUILDING FUTURE DATA FRAME PREDICTIONS SYNTAX >>>--------------------
st.subheader('Future Forecasted Price preview')
st.write(df_future)
fut_data=df_future.copy()
fut_data.reset_index('Date', inplace=True)
merged_df = pd.concat([df.set_index('Date'), fut_data.set_index('Date')], axis=1)
st.subheader('Final merged Data preview:')
st.write(merged_df)
merged_df.reset_index('Date', inplace=True)
merged_df['Date']=pd.to_datetime(merged_df['Date'])
merged_df.set_index('Date', inplace=True)

# ---------------------<<< DATAFRAME INFO SYNTAX >>>--------------------
# Create a buffer to capture the output of df.info()
# import io
# buffer = io.StringIO()
# merged_df.info(buf=buffer)
# info_str = buffer.getvalue()
# # Display the DataFrame info in Streamlit
# st.text("DataFrame Info:")
# st.text(info_str)

# ---------------------<<< Actual Price vs Future Forecasted Price >>>----------------------
st.subheader('Actual Price vs Future Forecasted Price')
fig = plt.figure(figsize=(10,4))
# plt.plot(merged_df[['Sulpur_rate','Predicted']])
plt.plot(merged_df)
plt.ylabel('rate of sulphur')
plt.xlabel('Date')
plt.legend(["actual rate of sulphur","Forecasted rate of sulphur"])
st.pyplot(fig)
# ---------------------<<< ---------------------------------------- >>>----------------------





