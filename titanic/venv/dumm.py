import pandas as pd
import io
import numpy as np
import math
filename='C:\\Users\\nikky\\Downloads\\train.csv'
db = pd.read_csv(filename, usecols=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
db1=pd.read_csv(filename, usecols=['Survived'])
dbt=pd.read_csv('C:\\Users\\nikky\\Downloads\\test.csv')
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
print("Y shape" +str(Y.shape))
X_train = db[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
print("X train shape"+str(X_train.shape))
w = np.zeros((X_train.shape[1], 1))
print("w shape"+str(w.shape))
b = 0
m=db.shape[0]
z1=np.dot( X_train,w)+b
print("z1"+str(z1))
A = (np.array(sigmoid(z1),dtype=np.float32));
a=np.log(A)
b1=np.log(1-A)
cost = (- 1 / m) * np.sum(Y * a + (1 - Y)*b1  )
dw = (1 / m) * np.dot(X_train, ((A - Y).T))
db1 = (1 / m) * np.sum(A - Y)
