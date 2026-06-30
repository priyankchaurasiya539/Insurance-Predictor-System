import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error
from xgboost import XGBRegressor


#Load the saved artifacts
artifacts = joblib.load("models/processed.pkl")
X_train = artifacts["X_train_encoded"]
X_test = artifacts["X_test_encoded"]
y_train = artifacts["y_train"]       
y_test  = artifacts["y_test"]

#Load the raw dataset
df_raw = pd.read_csv("data/insurance.csv")
y_raw = df_raw["charges"]

m , n , y_train , y_test = train_test_split(y_raw , y_raw ,test_size=0.25, random_state=42)

#Apply the linear regression model
regression = LinearRegression()
regression.fit(X_train , y_train)

#Apply the XGB Regressor model

xgb = XGBRegressor(
    n_estimators = 200 , 
    max_depth = 6 , 
    learning_rate = 0.05,
    random_state = 42
)
xgb.fit(X_train , y_train)

#predictions
y_pred = regression.predict(X_test)
y_pred_xgb = xgb.predict(X_test)


#score 
score = r2_score(y_test , y_pred)
mae = mean_absolute_error(y_test , y_pred)
print("R2_score_LR : " , round(score , 2))
print("MAE_LR : " , round(mae),2)

score_xgb = r2_score(y_test , y_pred_xgb)
mae_xgb = mean_absolute_error(y_test , y_pred_xgb)
print("R2_score_XGB : " , round(score_xgb , 2))
print("MAE_XGB : " , round(mae_xgb),2)


#Saving the model 

#as i have seen that when i apply LinearRegression Algo then my model shows 77% R2 score 
#and xgb regressor gives the 85% r2 score which is more reliable so we storing this pkl file

joblib.dump(xgb, "models/xgb_insurance.pkl")
print("XGBoost model saved → models/xgb_insurance.pkl")