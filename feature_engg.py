import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("data/insurance.csv")
print(df.head(20))
print(df.shape)
print(df.columns)
print(df.isnull().sum())        #Zero missing values
print(df.info())

#Perform data encoding to convert categorial values into numerical values
#first we do train_test_split


#Split the dependent and independent features
X = df.drop(columns=["charges"] , errors="ignore")
y = df["charges"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=42)

#Perform One Hot Encoding to convert categorial values to numerical values to train the model
ohe = OneHotEncoder()
ohe_cols = ["sex" , "smoker" , "region"]
X_train = pd.get_dummies(X_train , columns=ohe_cols , drop_first=True ,dtype=float)
X_test = pd.get_dummies(X_test , columns= ohe_cols , drop_first=True , dtype=float)

#Alignment of columns 
X_train , X_test = X_train.align(X_test , join="left" , axis=1  , fill_value=0.0)


#Converting whole dataset into float
X_train = X_train.astype(float)
X_test = X_test.astype(float)


#Saving the artifacts 
pipeline_artifacts = {
    "X_train_encoded" : X_train,
    "X_test_encoded" : X_test,
    "y_train" : y_train ,
    "y_test" : y_test

}
joblib.dump(pipeline_artifacts , "models/processed.pkl")

print("Successfully saved the file.")
print(X_train.info())