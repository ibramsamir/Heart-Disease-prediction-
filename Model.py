import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Reading data and displaying basic information
df = pd.read_csv(r'G:\NTI AI\NTI Project\MODEL1\heart_disease.csv')
print(df.describe)


# Define the continuous features
continuous_features = ['age', 'cigsPerDay', 'diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']
print(df.shape)
df.info()
df=df.dropna()


# Convert categorical columns to numerical using label encoding
df['Heart_ stroke']=pd.Categorical(df['Heart_ stroke'])
df['Heart_ stroke']=df['Heart_ stroke'].cat.codes

df['Gender']=pd.Categorical(df['Gender'])
df['Gender']=df['Gender'].cat.codes

df['education']=pd.Categorical(df['education'])
df['education']=df['education'].cat.codes

df['prevalentStroke']=pd.Categorical(df['prevalentStroke'])
df['prevalentStroke']=df['prevalentStroke'].cat.codes


# Assuming 'education' is the only column you want to one-hot encode
df_encoded = pd.get_dummies(df, columns=['education'], drop_first=False)

# Convert DataFrame to NumPy array
df_array = df.to_numpy()

# Split data into features (X) and target variable (y)
X = df_array[:, 1:15]
y = df_array[:, 15]

for i, col in enumerate(continuous_features):
    # Display unique values in each categorical column
    unique_values = df[col].unique()
    print(f"\nUnique values in {col} (Column {i+1}/{len(continuous_features)}):")
    print(unique_values)

    # Plot bar chart for each categorical column
    plt.figure(figsize=(12, 6))
    sns.countplot(x=col, data=df)
    plt.title(f"Bar chart for {col}")
    plt.show()


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Machine Learning Models

# 1. Logistic Regression Model
lr_model = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), LogisticRegression(penalty='l2', C=12, max_iter=1000) )
lr_model.fit(X_train, y_train)
lr_model.score(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc_score = accuracy_score(y_test, lr_pred)
print("Logistic Regression Model Accuracy:", lr_acc_score)
print(classification_report(y_test, lr_pred))
ConfusionMatrixDisplay.from_estimator(lr_model, X_test, y_test)
plt.show()

# 2. Decision Tree Model
dt_model = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), DecisionTreeClassifier())
dt_model.fit(X_train, y_train)
dt_model.score(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc_score = accuracy_score(y_test, dt_pred)
print("Decision Tree Model Accuracy:", dt_acc_score)
print(classification_report(y_test, dt_pred))
ConfusionMatrixDisplay.from_estimator(dt_model, X_test, y_test)

# 3. Random Forest Model
rf_model = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), RandomForestClassifier(n_estimators=500))
rf_model.fit(X_train, y_train)
rf_model.score(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc_score = accuracy_score(y_test, rf_pred)
print("Random Forest Model Accuracy:", rf_acc_score)
print(classification_report(y_test, rf_pred))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)


# 4. Support Vector Machine
svc = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), SVC())
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_acc_score = accuracy_score(y_test, svc_pred)
print("Support Vector Machine Accuracy:", svc_acc_score)
print(classification_report(y_test, svc_pred))
ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test)

# 5. Naive Bayes Model
nb_model = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), GaussianNB())
nb_model.fit(X_train, y_train)
nb_model.score(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc_score = accuracy_score(y_test, nb_pred)
print("Naive Bayes Model Accuracy:", nb_acc_score)
print(classification_report(y_test, nb_pred))
ConfusionMatrixDisplay.from_estimator(nb_model, X_test, y_test)

# 6. k-Nearest Neighbors (KNN) Model
knn_model = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(), KNeighborsClassifier(n_neighbors=5))
knn_model.fit(X_train, y_train)
knn_model.score(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc_score = accuracy_score(y_test, knn_pred)
print("KNN Model Accuracy:", knn_acc_score)
print(classification_report(y_test, knn_pred))
ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)


# Function to evaluate model and return accuracy score
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy * 100

# List of models and their corresponding names
models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), KNeighborsClassifier()]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'Naive Bayes', 'k-Nearest Neighbors (KNN)']

# Evaluating models and creating a DataFrame
model_ev = pd.DataFrame({'Model': model_names, 'Accuracy': [evaluate_model(model, X_train, X_test, y_train, y_test) for model in models]})

# Plotting the bar chart to compare model accuracies
colors = ['red', 'green', 'blue', 'gold', 'black', 'purple']
plt.figure(figsize=(12, 5))
plt.title("Barplot Representing Accuracy of Different Models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_ev['Model'], model_ev['Accuracy'], color=colors)
plt.show()

# Conclusion
print("The Logistic Regression Model gives the best accuracy.")