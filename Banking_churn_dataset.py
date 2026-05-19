import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

df = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\Churn_Modelling.csv")
df.head()

df.info()
df.shape
df.isnull().sum()

X = df.drop(['Surname','Geography','Gender','Exited'],axis = 1)
y = df["Exited"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name,model in models.items:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(name, 'Accuracy:', acc)

print("Confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print('classification_report:\n',classification_report(y_test,y_pred))

df['Gender'].value_counts().plot(kind='bar')
pd.crosstab(df['Exited'],df['Gender']).plot(kind='bar')