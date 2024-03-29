from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle

label_map = {'idle':0,'holding_controller':1, 'texting':2, 'stretch':3, 'left_raise_full':4, 'left_raise_half':5, 'right_raise_full':6, 'right_raise_half':7, 'left_hand_on_desk':8, 'right_hand_on desk':9, 'talking_left':10, 'talking_right':11}
actions = np.array(['idle','holding_controller', 'texting', 'stretch', 'left_raise_full', 'left_raise_half', 'right_raise_full', 'right_raise_half', 'left_hand_on_desk', 'right_hand_on desk', 'talking_left', 'talking_right'])
sequence_length = 300

df = pd.read_csv('coords.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

for algo,model in fit_models.items():
    y_hat = model.predict(X_test)
    print(algo, accuracy_score(y_test,y_hat))

with open('pose_model.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'],f)

