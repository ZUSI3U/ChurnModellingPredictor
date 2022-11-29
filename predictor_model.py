import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier # CatBoost
from imblearn.over_sampling import SMOTE

"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
"""

df = pd.read_csv(r"app/Include/Churn_Modelling.csv").copy()

X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

geography = pd.get_dummies(prefix='Geo', data=df['Geography'], drop_first=True)
gender = pd.get_dummies(df['Gender'], drop_first=True)

X = pd.concat([X, geography, gender], axis=1)
X = X.drop(['Geography', 'Gender'], axis=1)


SC = StandardScaler()
X = SC.fit_transform(X)

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

model = CatBoostClassifier()
model.fit(X_train, y_train, silent=True)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

model.save_model(r"app/catboost_model/model.json",format="json")
print("Saved model to disk")

"""
classifier = Sequential()

# bementi reteg es az elso rejtett reteg
classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))
classifier.add(Dropout(0.1))

# masodik rejtett reteg
classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu'))
classifier.add(Dropout(0.1))

# kimeneti reteg
classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Adamax

classifier.fit(X_train, y_train, validation_split=0.33, batch_size = 10, epochs = 150) # validation_split=0.33 epochs = 100

loss, acc = classifier.evaluate(X_test, y_test, verbose=0)
print("Model loss: %.2f, Accuracy: %.2f" % ((loss*100), (acc*100)))

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

score = accuracy_score(y_test, y_pred)

print("acc_score: ", score)


model_json = model.to_json()
with open(r"app/catboost_model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"app/catboost_model/model.h5")
print("Saved model to disk")
"""
#classifier.save(r"app/ann_model")
