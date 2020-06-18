import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("glass.csv")
x_train = df.drop("Type",axis=1)
y_train = df["Type"]

x_train, x_test, y_train, y_test= train_test_split(x_train, y_train, test_size=0.3, random_state=0)

#Using Bayle's Algorithm
guassian = GaussianNB()
guassian.fit(x_train,y_train)
y_pred = guassian.predict(x_test)
guassian_score = round(guassian.score(x_test,y_test) * 100,2)

print("Accuracy Score:",guassian_score)
print("Classification Report:\n",classification_report(y_test,y_pred))