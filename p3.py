import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Dataset
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','No'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes'],
    ['Overcast','Hot','High','Weak','Cool','Same','Yes'],
    ['Rainy','Warm','Normal','Weak','Warm','Change','No'],
    ['Sunny','Hot','Normal','Strong','Warm','Same','Yes'],
    ['Rainy','Warm','High','Strong','Cool','Change','No'],
    ['Overcast','Warm','Normal','Weak','Warm','Same','Yes'],
    ['Sunny','Cold','Normal','Weak','Cool','Same','Yes'],
    ['Rainy','Hot','High','Strong','Warm','Change','No'],
    ['Sunny','Hot','High','Weak','Warm','Same','No'],
    ['Overcast','Cold','Normal','Strong','Cool','Same','Yes'],
    ['Rainy','Warm','High','Weak','Cool','Change','No'],
    ['Sunny','Hot','Normal','Strong','Cool','Same','Yes']
]

features = ['Outlook','Temperature','Humidity','Wind','Water','Forecast']
df = pd.DataFrame(data, columns=features+['EnjoySport'])

# Encode string values to numbers
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split features/target
X = df[features]
y = df['EnjoySport']

# Train Decision Tree
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Show tree
print(export_text(model, feature_names=features))

# Predict new sample (encoded manually)
new_sample = [['Sunny','Hot','Normal','Strong','Cool','Same']]
new_df = pd.DataFrame(new_sample, columns=features)
for col in new_df.columns:
    new_df[col] = le.fit_transform(new_df[col])  # re-encode

print("Prediction:", model.predict(new_df)[0])
