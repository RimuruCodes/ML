import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')




label_encoder = LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'])
df['SeniorCitizen'] = label_encoder.fit_transform(df['SeniorCitizen'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure'] = label_encoder.fit_transform(df['tenure'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = label_encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = label_encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = label_encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = label_encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = label_encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = label_encoder.fit_transform(df['StreamingMovies'])
df['Contract'] = label_encoder.fit_transform(df['Contract'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = label_encoder.fit_transform(df['PaymentMethod'])
df['MonthlyCharges'] = label_encoder.fit_transform(df['MonthlyCharges'])
df['Churn'] = label_encoder.fit_transform(df['Churn'])

print(df['gender'].unique())
print(df['SeniorCitizen'].unique())
print(df['Partner'].unique())
print(df['Dependents'].unique())
print(df['TotalCharges'].unique())
print(df['tenure'].unique())
print(df['PhoneService'].unique())
print(df['MultipleLines'].unique())
print(df['InternetService'].unique())
print(df['OnlineSecurity'].unique())
print(df['OnlineBackup'].unique())
print(df['DeviceProtection'].unique())
print(df['TechSupport'].unique())
print(df['StreamingTV'].unique())
print(df['StreamingMovies'].unique())
print(df['Contract'].unique())
print(df['PaperlessBilling'].unique())
print(df['PaymentMethod'].unique())
print(df['MonthlyCharges'].unique())
print(df['Churn'].unique())


y = df['Churn']
X = df.drop(columns=['Churn', 'customerID'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = xgb.XGBClassifier(
    objective="binary:logistic",               
    eval_metric="mlogloss",     
    learning_rate=0.1,
    max_depth=4,
    n_estimators=100,
    random_state=42
)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


custom=pd.DataFrame([{
    'gender': 0,           
    'SeniorCitizen': 0,
    'Partner': 0,          
    'Dependents': 0,
    'tenure': 1,           
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 1,  
    'OnlineSecurity': 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'Contract': 0,         
    'PaperlessBilling': 1,
    'PaymentMethod': 2,    
    'MonthlyCharges': 75.5,
    'TotalCharges': 75.5
}])

prediction = model.predict(custom)
probability = model.predict_proba(custom)

print(f"Will they churn? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Certainty: {probability[0][prediction[0]] * 100:.2f}%")