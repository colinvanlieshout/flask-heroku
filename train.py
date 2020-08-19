import pandas as pd
from sklearn.linear_model import LogisticRegression
# create df
train = pd.read_csv('titanic.csv') # change file path
# drop null values
train.dropna(inplace=True)
# features and target
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']
# X matrix, y vector
X = train[features]
y = train[target]
# model
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)

#store the model in a pickle file
import pickle
pickle.dump(model, open('model.pkl', 'wb'))