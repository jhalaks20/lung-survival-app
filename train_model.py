import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv(r"C:\Users\jssm1\Downloads\dtset.csv")
df.columns = df.columns.str.strip()

df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['cholesterol_level'].fillna(df['cholesterol_level'].median(), inplace=True)

df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], dayfirst=True, errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], dayfirst=True, errors='coerce')
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

df.dropna(subset=['diagnosis_date', 'end_treatment_date'], inplace=True)

binary_cols = ['gender', 'family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'survived']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

if 'country' in df.columns:
    top_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

df = pd.get_dummies(df, columns=['country', 'smoking_status', 'cancer_stage', 'treatment_type'], drop_first=True)

df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'], inplace=True, errors='ignore')

X = df.drop('survived', axis=1)
y = df['survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)

joblib.dump(model, "lung_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model, scaler, and feature list saved successfully.")