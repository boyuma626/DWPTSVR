import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split 
import random
from gurobipy import *
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
from scipy.stats import uniform, randint
def hvec(A):
    m = A.shape[0]
    vec = np.zeros(int(m*(m+1)/2))
    index = 0

    for i in range(m):
        for j in range(i,m):
            vec[index] = A[i][j]
            index += 1

    return vec
def lvec(x):
    N, n = x.shape
    s = []
    for i in range(N):
        for j in range(n):
            for k in range(j, n):
                if k == j:
                    c = 1 / 2 * x[i, j] * x[i, j]
                    s.append(c)
                else:
                    c = x[i, j] * x[i, k]
                    s.append(c)

    s = np.array(s).reshape(N, -1)

    return s
def generateG(x):
    N, n = x.shape
    e = np.ones((N, 1))
    s = lvec(x)
    z = np.hstack((s, x, e))
    eta = lvec(z)
    r = np.hstack((eta, s))
    G = np.hstack((r, x, e))
    return G
def get_Hk(k):

    I_k = np.eye(k)
    I_=np.eye(int(k*(k+1)/2))#I_k(k+1)/2
    diag_I_k = np.diag(hvec(I_k))
    Hk = 2 * I_ - diag_I_k
    
    return Hk
def generateH(x):
    N, n = x.shape
    N,n=int(N),int(n)
    nn=n*(n+1)/2
    L=int((n*(n+1))/2+n+1)
    LL=L*(L+1)/2
    H_m=get_Hk(n)
    H_L=get_Hk(L)
    H = np.block([[H_L, np.zeros((int(LL), int(nn)))],
              [np.zeros((int(nn),int(LL))), H_m]])

    return H
    
def generater(x):
    N, n = x.shape
    e = np.ones((N, 1))
    s = lvec(x)
    z = np.hstack((s, x, e))
    eta = lvec(z)
    r = np.hstack((eta, s))
    return r
    
 

class DWPSVRP:
    """
    :parameter
    C1:
    C2:
    nu1:
    lambda1:
    """
    def __init__(self,C1,eps):
        self.C1 = C1
        self.eps = eps
     
        self.u1 = None
        self.u2 = None
        self.x_tr = None
        self.y_tr = None
        self.y_tr_predict = None
        self.x_te = None
        self.y_te = None
        self.y_te_predict = None
    
    def fit(self, x_tr, y_tr):#应该可以用输入x替代G
        C1=self.C1 
        eps=self.eps
        

        H  =generateH(x_tr)
        # diagonal = np.diag(H)
        diag_inv=np.diag(np.linalg.inv(H)).reshape(-1, 1)
        diag_inv_root = np.sqrt(diag_inv)
        
        r=generater(x_tr)
      

        N, n = x_tr.shape
        v1,v2=r.shape
        L=int((n*(n+1))/2+n+1)
        LL=L*(L+1)/2+(n*(n+1))/2
        LL=int(LL)
        G_tr= generateG(x_tr)
        G = np.hstack((r, x_tr))
        
        e = np.ones((N, 1))
        #v = cp.Variable((v2, 1))
        #b = cp.Variable((n, 1))
        u1=cp.Variable((v2+n, 1))
        q = cp.Variable((1, 1))
        xi1=cp.Variable((N, 1))
        xi11=cp.Variable((N, 1))
        
        #expr1 =0.5*(b.T@b)
        
        H=np.block([[H, np.zeros((int(LL), n))],
              [np.zeros((n,int(LL))), np.ones((n,n))]])
        H=cp.atoms.affine.wraps.psd_wrap(H)
       
   
        #objective = cp.Minimize( 0.5*expr2.T@expr11@expr2+q.T @ expr2+expr3)
        objective = cp.Minimize((1/2)*cp.quad_form(u1, H) + C1*(e.T@(xi1+xi11))) 
        constraints = [ G@u1+q-y_tr<=eps*e+xi11 , y_tr-G@u1-q<=eps*e+xi1, 0 <= xi11, 
                       0 <= xi1]
        prob = cp.Problem(objective, constraints)
        results = prob.solve(solver=cp.GUROBI)
        #results = prob.solve(solver=cp.GUROBI)

        #obj_val = []
        #obj_val.append(prob.objective.value)

        # 求解TWDWPSVR-2对偶问题

        # 指明参数有助于调节参数以达到最优
       
       
        #print(alpha11)
        #v=np.transpose((a*r*diag_inv.T))@e
        #v=v.value
        #diag_inv_1=diag_inv.T*np.ones((7,7))
        #v=diag_inv_1@r.T@a
        #print('v',v)
        #b=np.transpose((a*x_tr))@e
        #b=b.value
        #b=x_tr.T@a
        #print('b',b)
        u1=u1.value
        q=q.value
        

        #print(sum(q) / len(q)   )      
        self.u1 = u1
        #self.b = b
        self.q = q   
       
        
        # 计算error
        rmse_tr = 0
        mae_tr = 0
        self.rmse_tr = rmse_tr
        self.mae_tr = mae_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        
    def predict(self, x_te): #仅需要x_tr去训练
        u1=self.u1
        q=self.q
        G_te= generateG(x_te)
        u1 = np.vstack((u1, q))
        #print('u1',u1)
        #print(G_te.shape)
        #print(u1.shape)
        y_hat=G_te@u1
        self.y_te_predict = y_hat
        
        rmse_te = 0
        self.rmse_te = rmse_te
        #mae_te = 0
        #self.mae_te = mae_te
        
        self.x_te = x_te
        #self.y_te = y_te
        return y_hat
    def random_search0(X_train, X_test, y_train, y_test, n_iter=5):
        param_={    'C1': [2**i for i in range(-8, 9)],       
                        'eps': [2**i for i in range(-3, 4)], 
                }
        


   
  
        n = len(param_)
        dict={}
        def get_list(n):
            list=[0*i for i in range(2**n)]
            i1=len(list)//2
            #i2=2*i1
            for j in range(i1,2**n):
                list[j]=1
            # for j in range(i2,2**n):
            #     list[j]=2
            return list
        for i in range(2 ** n):
            dict[f"dict{i}"] = {}
        for j,key in enumerate(param_):
            length = len(param_[key])
            # split_idx = length // 2
            s1 = length // 2
            # s2=2*s1
            
            for i in range(2**n):    
                if (2**(n-j-1)*get_list(j+1))[i] == 0:
                # if 
                    half = param_[key][:s1]
                # elif  (2**(n-j-1)*get_list(j+1))[i] == 1:
                #     half = param_space[key][s1:s2]
                else:
                    half=param_[key][s1:]
                dict[f"dict{i}"].update({key: half})
        best_score = float('inf')
        best_params = []

        for _ in range(n_iter):

            params = {}
            #values=[1,1,0.5,0.5]
            for i in range(2**n):
                for key, value in dict[f"dict{i}"].items():
                    params[key] = random.choice(value)
                #这里的random要是有规则的random，以便更好的找到表现好的解
                
                           
                values = list(params.values())
                # print(values)
                reg=DWPSVRP(values[0], values[1])
                # print(values[0], values[1],values[2], values[3])
                reg.fit(X_train,y_train)
                y_pred=reg.predict(X_test)   
                score = mean_squared_error(y_test, y_pred, squared=False)+mean_absolute_error( y_test,y_pred)
            
                if score < best_score:
                    best_score = score
                    best_params = values
      
        return best_params
