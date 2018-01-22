import pandas as pd
import io
import numpy as np
import math
from kk import *
filename='C:\\Users\\nikky\\Downloads\\kaggle\\train.csv'
db = pd.read_csv(filename, usecols=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
db1=pd.read_csv(filename, usecols=['Survived'])
dbt=pd.read_csv('C:\\Users\\nikky\\Downloads\\kaggle\\test.csv')
dbt1=pd.read_csv('C:\\Users\\nikky\\Downloads\\kaggle\\gender.csv', usecols=['Survived','PassengerId'])


db.Embarked=db['Embarked'].fillna('C')
db.Embarked[db.Embarked == 'S'] = 1
db.Embarked[db.Embarked == 'C'] = 2
db.Embarked[db.Embarked == 'Q'] = 3
db.Age=db['Age'].fillna(np.mean(db.Age))
db['Age'][db['Age']<=18]=0
db['Age'][db['Age']>=18]=1
db['Sex'][db['Sex']=='male']=0
db['Sex'][db['Sex']=='female']=1

dbt.Embarked=dbt['Embarked'].fillna('C')
dbt.Embarked[dbt.Embarked == 'S'] = 1
dbt.Embarked[dbt.Embarked == 'C'] = 2
dbt.Embarked[dbt.Embarked == 'Q'] = 3
dbt.Age=db['Age'].fillna(np.mean(dbt.Age))
dbt.Fare=db['Fare'].fillna(np.mean(dbt.Fare))
dbt['Age'][dbt['Age']<=18]=0
dbt['Age'][dbt['Age']>=18]=1
dbt['Sex'][dbt['Sex']=='male']=0
dbt['Sex'][dbt['Sex']=='female']=1


def sigmoid(z):

  s = 1 / (1 + np.exp((-1*z.astype(float))))
  ### END CODE HERE ###

  return s
Y = db1.Survived.values
Y=Y.reshape(db1.shape[0],1)
X_train = db[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
X_test=dbt[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
Y1 = dbt1.Survived.values
Y1=Y1.reshape(Y1.shape[0],1)

w = np.zeros((X_train.shape[1], 1))
b = 0


parameters, grads, costs = optimize(w, b, X_train, Y, num_iterations=10000, learning_rate=0.01, print_cost=False)

w = parameters["w"]
b = parameters["b"]

Y_prediction_train = predict(w, b, X_train)
Y_prediction_test=predict(w, b, X_test)
output = pd.DataFrame(columns=['PassengerId', 'Survived'])
passId = dbt1["PassengerId"].values
output['PassengerId'] = passId
output['Survived'] = Y_prediction_test.astype(int)
output.to_csv('logisticRegressionSubmit.csv', index=False)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y1)) * 100))

