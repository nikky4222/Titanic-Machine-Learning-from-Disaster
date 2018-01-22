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
X_train = db[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
w = np.zeros((X_train.shape[1], 1))
print("w shape"+str(w.shape))
b = 0


def propagate(w, b, X_train, Y):
  m=db.shape[0]
  z1=np.dot(X_train,w)+b
  A = (np.array(sigmoid(z1),dtype=np.float32));
  a=np.log(A)
  b1=np.log(1-A)
  cost = (- 1 / m) * np.sum(Y * a + (1 - Y)*b1  )
  dw = (1 / m) * np.dot(X_train.T, ((A - Y)))
  db1 = (1 / m) * np.sum(A - Y)

  grads = {"dw": dw,
             "db": db1}
  return grads, cost


def optimize(w, b, X_train, Y, num_iterations, learning_rate, print_cost=False):


  costs = []

  for i in range(num_iterations):

    # Cost and gradient calculation (≈ 1-4 lines of code)
    ### START CODE HERE ###
    grads, cost = propagate(w, b, X_train, Y)
    ### END CODE HERE ###

    # Retrieve derivatives from grads
    dw = grads["dw"]
    db = grads["db"]

    # update rule (≈ 2 lines of code)
    ### START CODE HERE ###
    w = w - learning_rate * dw  # need to broadcast
    b = b - learning_rate * db
    ### END CODE HERE ###

    # Record the costs
    if i % 100 == 0:
      costs.append(cost)

    # Print the cost every 100 training examples
    if print_cost and i % 100 == 0:
      print("Cost after iteration %i: %f" % (i, cost))

  params = {"w": w,
            "b": b}

  grads = {"dw": dw,
           "db": db}
  return params, grads, costs

def predict(w, b, X_train):
  m = X_train.shape[0]
  Y_prediction = np.zeros((m,1))
  A = sigmoid(np.dot(X_train,w) + b)
  for i in range(A.shape[0]):
    var1 = A[i, 0] > 0.5
    if var1:
      Y_prediction[i, 0] = 1
    else:
      Y_prediction[i, 0] = 0
  print(Y_prediction)
  print("--------------------")

  return Y_prediction
