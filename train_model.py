import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
fish_data = pd.read_csv('Fish.csv')

# Splitting the data into features and target variable
X = fish_data.drop(columns=['Species', 'Weight'])
y = fish_data['Weight']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training a basic Random Forest Regressor model
basic_rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
basic_rf_model.fit(X_train, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(basic_rf_model, f)

