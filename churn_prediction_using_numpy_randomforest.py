import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

X = np.array([
    # [Monthly Charges, Tenure, Support Calls, Contract Type]
    [50, 2, 5, 0],
    [70, 12, 1, 1],
    [60, 5, 3, 0],
    [90, 24, 0, 1],
    [55, 3, 4, 0],
    [100, 36, 0, 1],
    [45, 1, 6, 0],
    [85, 20, 1, 1],
    [65, 6, 2, 0],
    [110, 40, 0, 1]
])
y = np.array([1,0,1,0,1,0,1,0,1,0])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Prediction",prediction)
print("Accuracy:",accuracy_score(y_test,prediction))
print("Confusion Matrix:\n",confusion_matrix(y_test,prediction))

new = [[54,1,5,0]]
result = model.predict(new)

print("Churn" if result[0]==1 else "Staying")