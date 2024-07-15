import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split 

from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
from scipy.stats import uniform
import pandas as pd
from sklearn.model_selection import train_test_split 
import math
import time

from sklearn.svm import SVR


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import pickle
class training:
    
        
    def __init__(self, x,y):
        self.x = x
        self.y = y
         

    def fit(self):
      
        
        x = self.x
        y = self.y

        lst=[]
        np.random.seed(1) # 设置种子为 0
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        a = 2*(np.percentile(y_train,95)-np.percentile(y_train,5))
        for i in range(len(y_train)):
            if i >= 0.8*len(y_train):
                if np.random.random(1)>=0.5:
                    y_train[i] += np.random.normal(a,0.5)
                else:
                    y_train[i] -= np.random.normal(a,0.5)
        xx_te=list(X_test.flatten())
        yy_te=list(y_test.flatten())
        xx_tr=list(X_train.flatten())
        yy_tr=list(y_train.flatten())
        for _ in range(10):

            


            model1 = SVR(kernel='poly')

            # 定义随机搜索超参数空间
            param_dist = {
                'C': uniform(2 ** np.random.randint(-8, 9)),
                'gamma':  uniform(2 ** np.random.randint(-3, 4)),
                'degree': randint(low=1, high=5),

                # 核系数
                # 惩罚系数
            }

            # 定义随机搜索器
            random_search = RandomizedSearchCV(model1, param_distributions=param_dist,
                                            n_iter=100, cv=5, random_state=42, n_jobs=-1)
            random_search.fit(X_train, y_train.ravel())
            best_params1=list(random_search.best_params_.values())


            model_svr = SVR(kernel='linear')
            param_dist1 = {'C': uniform(2 ** np.random.randint(-8, 9)),}

            # 使用随机搜索进行超参数调节
            random_search = RandomizedSearchCV(model_svr, param_distributions=param_dist1, n_iter=10, cv=5, random_state=42, n_jobs=-1)
            random_search.fit(X_train, y_train.ravel())
            best_params2=list(random_search.best_params_.values())

            
            # 划分训练集和测试集


            from model.DWPTSVR import DWPTSVR
            from model.QSTSVR import QSTSVR
            from model.rbfTSVR import rbfTSVR
            from model.TSVR import TSVR

            para_01=rbfTSVR.random_search_1(x,y)
            
            reg01=rbfTSVR(para_01[0],para_01[0],para_01[1],para_01[1],para_01[2],para_01[2])

            start = time.time()  # 记录开始时间 
            reg01.fit(X_train,y_train)
            end=time.time()
            t01 = end- start

            y_hat01=reg01.predict(X_train,X_test)
            y_hat_01=reg01.predict(X_train,X_train)#_代表是train
            


            #参数空间：14，000

            para_00=TSVR.random_search(x,y)
             # 记录开始时间 
            reg00=TSVR(para_00[0],para_00[1],para_00[2],para_00[3])

            start = time.time() 
            reg00.fit(X_train,y_train)
            end=time.time()
            t00 = end- start

            y_hat00=reg00.predict(X_test)
            y_hat_00=reg00.predict(X_train)
  
            #reg=DWPTSVR(8, 1, 0.1, 0.95)
            #参数空间：7*17*19*19=43，000

            para1=DWPTSVR.random_search1(x,y)
            para2=QSTSVR.random_search2(x,y)
            import matplotlib.pyplot as plt
            # 加载数据集

            #参数空间：17*7=119
            from model.DWPSVRP import DWPSVRP

            para=DWPSVRP.random_search0(x,y)

            # 加载数据集

              # 记录开始时间 
            reg0=DWPSVRP(para[0],para[1])

            start = time.time()
            reg0.fit(X_train,y_train)
            end=time.time()
            t0 = end- start

            y_hat0=reg0.predict(X_test)
            y_hat_0=reg0.predict(X_train)
            



              # 记录开始时间 
            reg1=DWPTSVR(16,para1[0],0.2,para1[1],16,para1[2],0.2,para1[3])
            start = time.time()
            reg1.fit(X_train,y_train)
            end=time.time()
            t1 = end- start


            y_hat1=reg1.predict(X_test)
            y_hat_1=reg1.predict(X_train)
            



            #同DWPTSVR：43，000
            
            reg2=QSTSVR(16,para2[0],0.2,para2[1],16,para2[2],0.2,para2[3])

            start = time.time()
            reg2.fit(X_train,y_train)
            end=time.time()
            t2 = end- start

            y_hat2=reg2.predict(X_test)
            y_hat_2=reg2.predict(X_train)
            


              # 记录开始时间
            reg3 = SVR(kernel='linear',C=best_params2[0])

            start = time.time()
            reg3.fit(X_train, y_train.ravel())
            end=time.time()
            t3 = end- start


            y_hat3 = reg3.predict(X_test)
            y_hat_3 = reg3.predict(X_train)
            


           


              # 记录开始时间
            reg5= SVR(kernel='poly',C=best_params1[0],degree=best_params1[1],gamma=best_params1[2],)

            start = time.time()
            reg5.fit(X_train, y_train.ravel())
            end=time.time()
            t5 = end- start

            y_hat5 = reg5.predict(X_test)
            y_hat_5 = reg5.predict(X_train)
            
            yy_hat0=list(y_hat0.flatten())
            yy_hat1=list(y_hat1.flatten())
            yy_hat2=list(y_hat2.flatten())
            yy_hat3=list(y_hat3.flatten())
            yy_hat5=list(y_hat5.flatten())
            yy_hat00=list(y_hat00.flatten())
            yy_hat01=list(y_hat01.flatten())

            yy_hat_0=list(y_hat_0.flatten())
            yy_hat_1=list(y_hat_1.flatten())
            yy_hat_2=list(y_hat_2.flatten())
            yy_hat_3=list(y_hat_3.flatten())
            yy_hat_5=list(y_hat_5.flatten())
            yy_hat_00=list(y_hat_00.flatten())
            yy_hat_01=list(y_hat_01.flatten())
            df0=pd.DataFrame({
                    
                    'X_test':xx_te,
                    'y_test':yy_te,
                            'y_hat0':yy_hat0,
                                'y_hat1':yy_hat1,
                                'y_hat2':yy_hat2,
                                'y_hat3':yy_hat3,
                                #'y_hat4':yy_hat4,
                                'y_hat5':yy_hat5,
                                'y_hat00':yy_hat00,
                                'y_hat01':yy_hat01,
                                
                                
            }) 
            lst.append(df0)
            df1=pd.DataFrame({'X_train':xx_tr,
                                'y_train':yy_tr,
                                'y_hat_0':yy_hat_0,
                                'y_hat_1':yy_hat_1,
                                'y_hat_2':yy_hat_2,
                                'y_hat_3':yy_hat_3,
                                #'y_hat_4':yy_hat_4,
                                'y_hat_5':yy_hat_5,
                                'y_hat_00':yy_hat_00,
                                'y_hat_01':yy_hat_01,
            })
            lst.append(df1)
            df2=pd.DataFrame({'t0':[t0],
                                't1':[t1],
                                't2':[t2],
                                't3':[t3],
                                't5':[t5],
                                't00':[t00],
                                't01':[t01],
            })
            lst.append(df2)


            data_dict = {'para0':para,
                                'para1':para1,
                                'para2':para2,
                                'para3':best_params2,
                                'para5':best_params1,
                                'para00':para_00,
                                'para01':para_01,
            }
            df3 = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
            lst.append(df3)
        return lst




