import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("dtset.csv")
df.columns = df.columns.str.strip()

# Fill missing values
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['cholesterol_level'] = df['cholesterol_level'].fillna(df['cholesterol_level'].median())

# Convert date columns
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], dayfirst=True, errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], dayfirst=True, errors='coerce')
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

# Drop rows with missing dates
df.dropna(subset=['diagnosis_date', 'end_treatment_date'], inplace=True)

# Encode binary columns
binary_cols = ['gender', 'family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'survived']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Simplify country column to avoid high cardinality
if 'country' in df.columns:
    top_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

# One-hot encode multi-class columns
df = pd.get_dummies(df, columns=['country', 'smoking_status', 'cancer_stage', 'treatment_type'], drop_first=True)

# Drop unused ID/date columns
df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'], errors='ignore', inplace=True)

# Separate features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate class imbalance ratio
count_0 = y.value_counts()[0]  # Non-survivors
count_1 = y.value_counts()[1]  # Survivors
imbalance_ratio = count_0 / count_1

# Define and train model with imbalance handling
model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=imbalance_ratio
)

model.fit(X_scaled, y)

# Save model, scaler, and feature names
joblib.dump(model, "lung_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model trained successfully with class imbalance handled.")

joblib.dump(model, "lung_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model, scaler, and feature list saved successfully.")
