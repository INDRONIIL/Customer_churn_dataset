import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\customer_churn_dataset-testing-master.csv")
df.head()

df.isnull().sum()
df['Gender']=df['Gender'].map({'Male':1,'Female':0})


# Step 2: Features and Target
X = df.drop(["Churn",'Subscription Type','Contract Length'], axis=1)
y = df["Churn"]


# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name,model in models.items:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(name, 'Accuracy:', acc)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))