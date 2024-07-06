import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split 
import random
from gurobipy import *
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
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
    
 

class DWPSVR:
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
        L=int((n*(n+1))/2+n+1)
        LL=L*(L+1)/2+(n*(n+1))/2
        LL=int(LL)
        
        
        
        e = np.ones((N, 1))
        alpha1 = cp.Variable((N, 1))
        alpha11 = cp.Variable((N, 1))
        
        expr1 =r@diag_inv_root@diag_inv_root.T@r.T+x_tr@x_tr.T
        expr11=cp.atoms.affine.wraps.psd_wrap(expr1)
        #expr11=np.block([[expr1, np.zeros((N,1))],
              #[np.zeros((1,N)), 0]])
        
        expr2 =  alpha11- alpha1
        expr3 = e.T@(alpha11+alpha1)*eps
        #expr4 = np.vstack((alpha11-alpha1,expr3))
        #p = np.vstack((-y_tr,np.ones((1,1))))
        qq=-y_tr
   
        #objective = cp.Minimize( 0.5*expr2.T@expr11@expr2+q.T @ expr2+expr3)
        objective = cp.Minimize((1/2)*cp.quad_form(expr2, expr11) + qq.T @ expr2+expr3)
        constraints = [0 <= alpha1, alpha1 <= C1, 0 <= alpha11, 
                       alpha11 <= C1,e.T@(alpha11- alpha1)==0]
        prob = cp.Problem(objective, constraints)
        results = prob.solve(solver=cp.GUROBI)
        #results = prob.solve(solver=cp.GUROBI)

        #obj_val = []
        #obj_val.append(prob.objective.value)

        # 求解TWDWPSVR-2对偶问题

        # 指明参数有助于调节参数以达到最优
        a=expr2.value
        alpha1=alpha1.value
        #print(alpha1)
        alpha11=alpha11.value
        #print(alpha11)
        #v=np.transpose((a*r*diag_inv.T))@e
        v=diag_inv*r.T@a
        #diag_inv_1=diag_inv.T*np.ones((7,7))
        #v=diag_inv_1@r.T@a
        #print('v',v)
        #b=np.transpose((a*x_tr))@e
        b=x_tr.T@a
        #b=x_tr.T@a
        #print('b',b)
        q=[]
        q_=0
        for i in range(len(x_tr)):
            #if alpha1[i] <=1e-5 and C1-1e-5<=alpha11[i]<=C1 :
            if alpha1[i] <=1e-5 and 1e-5<=alpha11[i] <=C1-1e-5:
                q_=y_tr[i]-eps-r[i]@v-x_tr[i]@b
                q.append(q_)
                
                 
            elif alpha11[i] <=1e-5 and 1e-5<=alpha1[i] <=C1-1e-5 :
                #q=y_tr[i]+eps-r[i]@v-x_tr[i]@b
                q_=y_tr[i]+eps-r[i]@v-x_tr[i]@b
                q.append(q_)  
            elif alpha1[i] <=1e-5 and C1-1e-5<=alpha11[i] <=C1+1e-5:
                q_=y_tr[i]-eps-r[i]@v-x_tr[i]@b
                q.append(q_)

        q = np.array(q)
        q=min(abs(q))  
        print('q',q) 

        #print(sum(q) / len(q)   )      
        self.v = v
        self.b = b
        self.q = q   
       
        
        # 计算error
        rmse_tr = 0
        mae_tr = 0
        self.rmse_tr = rmse_tr
        self.mae_tr = mae_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        
    def predict(self, x_te): #仅需要x_tr去训练
        v=self.v
        b=self.b
        q=self.q
        G_te= generateG(x_te)
        u1 = np.vstack((v, b, q))
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
    def random_search0(param_,x,y, n_iter=1):
        
        


        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
  
        best_score = float('inf')
        best_params = []

        for i in range(n_iter):
            params = {}
            values=[]
            for key, value in param_.items():
                params[key] = random.choice(value)
                
                          
            values = list(params.values())
            reg=DWPSVR(values[0], values[1])
            reg.fit(X_train,y_train)
            y_pred=reg.predict(X_test)   
            score = mean_squared_error(y_test, y_pred, squared=False)+mean_absolute_error( y_test,y_pred)
        
            if score < best_score:
                best_score = score
                best_params = values
      
        return best_params
