# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import required packages
2. read the dataset using pamdas as a data frame
3. compute cost values
4. Gradient Descent
![image](https://user-images.githubusercontent.com/116153626/230776444-811c5901-49bb-4fb0-90c7-a407ffabcc2f.png)
5.compute Cost function graph
6.compute prediction graph

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mohanapriya U
RegisterNumber: 212220040091
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('ex1.txt',header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of city (10,000s)")
plt.ylabel("profit ($10,000")
plt.title("profit prediction")
def compute(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
compute(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(compute(x,y,theta))
  return theta,j_history
theta,j_history=gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of city (10,000s)")
plt.ylabel("profit ($10,000")
plt.title("profit prediction")
def  predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))





## Ouput:

1.Profit Prediction Graph
![229281156-41c385f6-f6fe-45d1-80e3-203ac4d6fed0](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/6c97c297-947c-4f0b-9845-48e7798209c2)

2.Compute Cost Value
![229281208-965fc3f6-3837-4a44-9121-20c933053ca7](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/05c8b46b-d21e-4277-9dcd-079df03e6a13)

3.h(x) Value
![229281232-4a6580a3-4456-419d-83ed-eb8715622482](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/512ed170-c027-4160-92d4-cbd0bf767a34)

4.Cost function using Gradient Descent Graph
![229281240-a13c9f89-8b73-4663-ba05-5c2cef75c0e1](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/042eea27-4557-438f-a362-617a424f42bb)

5. Profit Prediction Graph
![229281252-e66d7906-c59b-4d4e-ad93-620bf44e71d9](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/5fd9dd83-7883-4bec-9abb-e77f9b39f3fb)

6.Profit for the Population 35,000
![229281259-eff7218b-8da4-4fac-813e-17dde6cc9368](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/52592112-0900-4576-887e-12c04c843c71)

7.Profit for the Population 70,000
![229281269-15a0f249-d347-4912-be8a-437cd504a8a1](https://github.com/MohanapriyaU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/116153626/3cd80d8f-fe76-4a1a-a10b-f9b666931931)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming


