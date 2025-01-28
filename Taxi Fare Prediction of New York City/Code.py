import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
print("All the libraries loaded successfully")

datatype = {"key":"object","fare_amount":"float64","pickup_datetime":"object",
            "pickup_longitude":"float64","pickup_latitude":"float64","dropoff_longitude":"float64","dropoff_latitude":"float64",
           "passenger_count":"int64"
           }
df = pd.read_csv('train.csv', nrows=100000,dtype=datatype, parse_dates=['pickup_datetime'])
# Assuming you have a test CSV file, adjust the path accordingly
test_df = pd.read_csv('test.csv')

# Extract features from pickup_datetime
df['pickup_day_of_week'] = df['pickup_datetime'].dt.day_name()
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_year'] = df['pickup_datetime'].dt.year

# Drop the original datetime column
df.drop(['key'],axis=1,inplace=True)
df.info()


print(df.head())

# Print the shape of the dataset
print("Shape of the dataset : ",df.shape)

# Visualize the fare amount
plt.figure(figsize = (12,5))
n, bins, patches = plt.hist(df.fare_amount,1000, facecolor="dodgerblue",alpha=0.75)
plt.xlabel("Fare Amount", fontsize=15,fontweight='bold')
plt.title("Histogram of Fare amount",fontsize=25,fontweight='bold')
plt.xlim(0,200)

import calendar  # Creating datetime features based on pickup_datetime

df['pickup_date'] = df['pickup_datetime'].dt.date
df['pickup_day'] = df['pickup_datetime'].apply(lambda x:x.day)
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x:x.hour)
df['pickup_day_of_week'] = df['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
df['pickup_month'] = df['pickup_datetime'].apply(lambda x:x.month)
df['pickup_year'] = df['pickup_datetime'].apply(lambda x:x.year)

print(df.head())

print(df.describe())   # describe the data

print(df.isnull().sum())  # Check for null values

#removing null values
print('old size: %d' % len(df))
train_df = df.dropna(how = 'any', axis = 'rows')
print('new size: %d' % len(train_df))

# Observe the change in the data
df = df[((df['pickup_longitude'] > - 78) &
         (df['pickup_longitude'] < -70)) &

        ((df['dropoff_longitude'] > -78) &
         (df['dropoff_longitude'] < -70)) &

        ((df['pickup_latitude'] > 37) &
         (df['pickup_latitude'] < 45)) &

        ((df['dropoff_latitude'] > 37) &
         (df['dropoff_latitude'] < 45)) &

        (df['passenger_count'] > 0) &
        (df['fare_amount'] >= 2.5)]

print(df.describe())

#calculate distance between two Geolocations
def hav_dist(lat1, lon1, lat2, lon2):
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = (np.sin(dLat / 2) ** 2 + np.sin(dLon / 2) ** 2 * np.cos(lat1) * np.cos(lat2))
    rad = 6371  # Earth's radius in kilometers
    c = 2 * np.arcsin(np.sqrt(a))

    distance = rad * c
    return distance
df['distance'] = hav_dist(df['pickup_latitude'], train_df['pickup_longitude'], df['dropoff_latitude'],
                                df['dropoff_longitude'])
df['distance'] = hav_dist(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'],
                               df['dropoff_longitude'])

print(df.head())

# Visualize the target variable
plt.figure(figsize=(8,5))
sns.kdeplot(np.log(df.fare_amount.values)).set_title((
    "Distribution of Fare in Log Scale"),fontsize=15, fontweight='bold')

# Analyzing the top 5 fare amount
df.fare_amount.nlargest(5)

df['fare_amount'].value_counts(normalize=True).iloc[:5]

# Visualize Passenger Count
plt.figure(figsize=(10,6))
df['passenger_count'].value_counts().plot.bar(color = 'dodgerblue', edgecolor='k')
plt.title("Histogram of Passenger Counts",fontsize=20,fontweight='bold')
plt.xlabel("Passenger Counts",fontsize=15,fontweight='bold')
plt.ylabel("Count",fontsize=15,fontweight='bold')

# Visualize Passenger Count
plt.figure(figsize=(10,6))
df['pickup_year'].value_counts().plot.bar(color = 'dodgerblue', edgecolor='k')
plt.title("Histogram of Pickup in Years",fontsize=20,fontweight='bold')
plt.xlabel("Pickup Counts",fontsize=15,fontweight='bold')
plt.ylabel("Count",fontsize=15,fontweight='bold')

# Heatmap for Pickups and dropoffs in NYC
city_long = (-74.03,-73.75)
city_lat = (40.63, 40.85)

df.plot(kind='scatter',x='dropoff_longitude',y='dropoff_latitude',color='dodgerblue',s=.02,alpha=.6)
plt.title("Dropoffs")
plt.ylim(city_lat)
plt.xlim(city_long)

# Heatmap for Pickups and dropoffs in NYC
city_long = (-74.03,-73.75)
city_lat = (40.63, 40.85)

df.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',color='green',s=.02,alpha=.6)
plt.title("Pickups")
plt.ylim(city_lat)
plt.xlim(city_long)
plt.show()

#Build, train and evalute the linear regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'fare_amount' is the target variable
y = df['fare_amount']
X = df.drop(['fare_amount'], axis=1)

# Extract features from pickup_datetime
X['pickup_day_of_week'] = X['pickup_datetime'].dt.day_name()
X['pickup_hour'] = X['pickup_datetime'].dt.hour
X['pickup_month'] = X['pickup_datetime'].dt.month
X['pickup_year'] = X['pickup_datetime'].dt.year

# Drop the original datetime column
X.drop(['pickup_datetime'], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
categorical_features = ['pickup_day_of_week', 'pickup_hour', 'pickup_month', 'pickup_year']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # You might need to import SimpleImputer
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numeric and categorical transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_test = model.predict(X_test)

# Evaluate the model using different metrics
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Create a DataFrame for visualization
visualization_df = pd.DataFrame({'Actual': y_test.values[:1000], 'Predicted': y_pred_test[:1000]})

# Scatter plot with regression line
plt.figure(figsize=(12, 8))
sns.regplot(x='Actual', y='Predicted', data=visualization_df, scatter_kws={'s': 20, 'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Linear Regression: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Fare Amount', fontsize=14)
plt.ylabel('Predicted Fare Amount', fontsize=14)
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Feature selection
features = ['distance', 'passenger_count']

# Select features and target variable
X = df[features]
y = df['fare_amount']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=42, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the differences between predicted and actual fares
differences = y_pred - y_test.values

# Evaluate the model using different metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualize the predictions against the actual fares
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Fare of Taxi')
plt.ylabel('Predicted Fare of Taxi')
plt.title('Neural Network Model: Actual vs Predicted Taxi Fares')
plt.show()