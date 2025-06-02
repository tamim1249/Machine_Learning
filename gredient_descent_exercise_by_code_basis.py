import numpy as np
import pandas as pd
df=pd.read_csv("test_scores.csv")
df
def gredient_descent(math,cs):
    m=b=0;
    n=len(math)
    itr=100
    learning_rate=0.0002
    for i in range(itr):
        y_pred=m*math+b
        cost=(1/n)*sum([val**2 for val in(cs-y_pred)])
        md=-(2/n)*sum(math*(cs-y_pred))
        bd=-(2/n)*sum(cs-y_pred)
        m=m-learning_rate*md
        b=b-learning_rate*bd
        print("m {} ,b {} , cost {}, itr {} ".format(m,b,cost,i))
math=np.array([92,56,88,70,80,49,65,35,66,67])
cs=np.array([92,68,81,80,83,52,66,30,68,73])
gredient_descent(math,cs)
