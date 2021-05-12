# have to say the ipynb is little inefficient

import pandas as pd
import numpy as np # use for the algebra operation
import matplotlib.pyplot as plt # use for visulisation
from matplotlib.pyplot import figure

X=np.array([[3,3],[4,3],[1,1]])
Y=np.array([1,1,-1])


class Model:
    def __init__(self,w=np.zeros(len(X)-1, dtype=np.float32),b=0,step=1):
        self.w = w
        self.b = b
        self.step=step;
        self.w_list=[np.copy(w)] #store the iteration of w
        self.b_list=[b]
        self.wrong_points = [0]
        print(self.w_list)
    def sign(self,x):
        y = np.dot(x,self.w)+self.b
        return y
    
    def fit(self,X_train,y_train):
        wrong_count = 1
        while  wrong_count:
            wrong_count = 0
            for X,y in zip(X_train,y_train):
                while  y * self.sign(X)<=0:
                    self.w +=self.step * np.dot(y,X)
                    self.b += self.step *y
                    self.w_list.append(np.copy(self.w))
                    self.b_list.append(self.b)
                    wrong_count +=1
        return  "Perceptron Model!"  

    def visualize_process(self):
        # fig, axs = plt.subplots(len(self.w_list))
        plt.rcParams["figure.figsize"] = (20,7)
        fig, axs = plt.subplots(2,4)
        fig.suptitle('the visualization of the process of the building model')
        a=0
        b=0
        for i in range(len(self.w_list)): 
            x_points = np.linspace(0,5,10)
            y_ = -(self.w_list[i][0] * x_points + self.b_list[i]) /self.w_list[i][1]
            axs[a,b].plot(x_points, y_,color='red')
            axs[a,b].plot(X[:2,0],X[:2,1],'bo',color='blue',label='1')
            axs[a,b].plot(X[2:3,0],X[2:3,1],'bo',color='orange',Label='-1')
            axs[a,b].axis([0,5,-2,4])
            a=int(i/4)
            b=i%4


perceptron = Model(w=np.zeros(len(X[0])))
perceptron.fit(X,Y)
perceptron.visualize_process()