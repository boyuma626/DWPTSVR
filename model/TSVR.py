import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split 
import random
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
def rbf_kernel(x, y, sigma):
    
    diff = np.linalg.norm(x - y) ** 2
    k = np.exp(-diff / (2 * sigma **2))
    return k
# 产生G矩阵
def generateG(x):
    N, n = x.shape
    e = np.ones((N, 1))
    
    G = np.hstack(( x, e))
    return G
class TSVR:
    
    
    def __init__(self, C1, C2,eps1,eps2):
        self.C1 = C1
        self.C2 = C2
        self.eps1=eps1
        self.eps2=eps2
        self.u1 = None#u1=[w1;b1]
        self.u2 = None#u2=[w2;b2]
        self.rmse_tr = None
        self.mape_tr = None
        self.rmse_te = None
        self.mape_te = None
        self.x_tr = None
        self.y_tr = None
        self.x = None
        self.y = None
        self.y_tr_predict = None
        self.x_te = None
        self.y_te = None
        self.y_te_predict = None

    def fit(self, x_tr, y_tr):#应该可以用输入x替代G
        C1 = self.C1
        C2 = self.C2     
        eps1=self.eps1
        eps2=self.eps2
        
        G_tr=generateG(x_tr)
        
        N, l = G_tr.shape
        
        e = np.ones((N, 1))
        f=y_tr-e*eps1
        alpha1 = cp.Variable((N, 1))
        
        expr1 = G_tr@np.linalg.inv(G_tr.T@G_tr)@G_tr.T
        expr1=expr1+0.001*np.identity(N, dtype=int)
        expr1 = np.triu(expr1)
        expr1 =     expr1 + expr1.T - np.diag(expr1.diagonal())
        
        #expr3 = cp.atoms.affine.wraps.psd_wrap(expr3)
        expr1 = cp.atoms.affine.wraps.psd_wrap(expr1)
        expr2=expr1@alpha1
        


        objective = cp.Minimize( f.T@alpha1- f.T @ expr2)
        #objective = cp.Minimize((1/2) * cp.quad_form(alpha1, expr1) - f.T @ expr2+f.T@alpha1)
        constraints = [0 <= alpha1, alpha1 <= C1 * e / N]
        prob = cp.Problem(objective, constraints)
        results = prob.solve(solver=cp.GUROBI)
        #results = prob.solve(solver='COPT')

        #obj_val = []
        #obj_val.append(prob.objective.value)

        # 求解TWDWPSVR-2对偶问题

        # 指明参数有助于调节参数以达到最优
        C2 = self.C2

    
        alpha2 = cp.Variable((N, 1))
        h=y_tr+e*eps2
        
        expr11 = G_tr@np.linalg.inv(G_tr.T@G_tr)@G_tr.T
        expr11=expr11+0.001*np.identity(N, dtype=int)
        expr11 = np.triu(expr11)
        expr11 =     expr11 + expr11.T - np.diag(expr11.diagonal())
        expr11 = cp.atoms.affine.wraps.psd_wrap(expr11)
        expr22=expr11@alpha2
        
       
        objective = cp.Minimize(h.T @ expr22-h.T@alpha2)
        #objective = cp.Minimize((1/2) * cp.quad_form(alpha2, expr11) +h.T @ expr22-h.T@alpha2)
        constraints = [0 <= alpha2, alpha2 <= C2 * e ]
        prob = cp.Problem(objective, constraints)
        results = prob.solve(solver=cp.GUROBI)
       
        alpha1=alpha1.value
        alpha2=alpha2.value

        u1=np.linalg.inv(G_tr.T@G_tr+0.0001*np.identity(l, dtype=int))@G_tr.T@(f-alpha1)
       
        u2=np.linalg.inv(G_tr.T@G_tr+0.0001*np.identity(l, dtype=int))@G_tr.T@(h+alpha2)
        
        self.u1 = u1
        self.u2 = u2
        
        # 计算error
        rmse_tr = 0
        mae_tr = 0
        self.rmse_tr = rmse_tr
        self.mae_tr = mae_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        
    def predict(self, x_te): #仅需要x_tr去训练
        G_te= generateG(x_te)
        u1 = self.u1
        u2 = self.u2
        y_hat1=G_te@u1
        y_hat2=G_te@u2
        y_hat=(y_hat1+y_hat2)/2
        self.y_te_predict = y_hat
        
        rmse_te = 0
        self.rmse_te = rmse_te
        #mae_te = 0
        #self.mae_te = mae_te
        
        self.x_te = x_te
        #self.y_te = y_te
        return y_hat


    #def random_search(param_space,x,y, n_iter=500):
    def random_search(x,y, n_iter=5):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=41)
        param_space={    'C1': [2**i for i in range(-8, 9)],
                'C2': [2**i for i in range(-8, 9)],
            'eps1': [2**i for i in range(-3, 4)], 
            'eps2': [2**i for i in range(-3, 4)],
              }    
        


       
  
        n = len(param_space)
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
        for j,key in enumerate(param_space):
            length = len(param_space[key])
            # split_idx = length // 2
            s1 = length // 2
            # s2=2*s1
            
            for i in range(2**n):    
                if (2**(n-j-1)*get_list(j+1))[i] == 0:
                # if 
                    half = param_space[key][:s1]
                # elif  (2**(n-j-1)*get_list(j+1))[i] == 1:
                #     half = param_space[key][s1:s2]
                else:
                    half=param_space[key][s1:]
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
                reg=TSVR(values[0], values[1],values[2], values[3])
                # print(values[0], values[1],values[2], values[3])
                reg.fit(X_train,y_train)
                y_pred=reg.predict(X_test)   
                score = mean_squared_error(y_test, y_pred, squared=False)+mean_absolute_error( y_test,y_pred)
            
                if score < best_score:
                    best_score = score
                    best_params = values
      
        return best_params
    
