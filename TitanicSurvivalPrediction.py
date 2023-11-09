import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('tested.csv')
data.head()

data = data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1)

data['Age'].fillna(data['Age'].median(), inplace = True)

labelEncoder = LabelEncoder()
data['Sex'] = labelEncoder.fit_transform(data['Sex'])

data.replace([np.inf, -np.inf], np.nan, inplace = True)

data.dropna(inplace = True)

a = data.drop('Survived', axis = 1)
b = data['Survived']

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 42)

dtc = DecisionTreeClassifier(random_state = 42)
dtc.fit(a_train, b_train)

b_prd = dtc.predict(a_test)

accuracy = accuracy_score(b_test, b_prd)
report = classification_report(b_test, b_prd)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classication Report:\n", report)
