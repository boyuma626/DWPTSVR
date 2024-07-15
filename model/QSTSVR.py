import numpy as np
# from coptpy import *
import cvxopt
import cvxpy as cp
from sklearn.model_selection import train_test_split 
import random
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
from scipy.stats import uniform, randint
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

def generateS(x):
    N, n = x.shape
    e = np.ones((N, 1))
    s = lvec(x)
    z = np.hstack((s, x))
            
    return z 
class QSTSVR:
    def generateS(x):
        N, n = x.shape
        e = np.ones((N, 1))
        s = lvec(x)
        z = np.hstack((s, x))
            
        return z
    
    def __init__(self, C1, C2, nu1, lambda1,C3, C4, nu2, lambda2):
        self.C1 = C1
        self.C2 = C2
        self.nu1 = nu1
        self.lambda1 = lambda1
        self.C3 = C3
        self.C4 = C4
        self.nu2 = nu2
        self.lambda2 = lambda2
        self.u1 = None
        self.u2 = None
        self.rmse_tr = None
        self.mape_tr = None
        self.rmse_te = None
        self.mape_te = None
        self.x_tr = None
        self.y_tr = None
        self.y_tr_predict = None
        self.x_te = None
        self.y_te = None
        self.y_te_predict = None

    def fit(self, x_tr, y_tr):#应该可以用输入x替代G
        C1 = self.C1
        C2 = self.C2
        nu1 = self.nu1
        lambda1 = self.lambda1
        
        S_tr=generateS(x_tr)

        N, l = S_tr.shape
        
        e = np.ones((N, 1))
        alpha1 = cp.Variable((N, 1))
        r1 = cp.Variable((N, 1))
        r11 = cp.Variable((N, 1))
        expr1 = 1 / (2 * lambda1)
        expr2 = lambda1 * y_tr - (alpha1 + (r11 - r1))
        expr3 = S_tr @ np.linalg.inv(S_tr.T @ S_tr + C1 * np.identity(l, dtype=int)) @ S_tr.T+0.001*np.identity(N, dtype=int)
        expr3 = np.triu(expr3)
        expr3 =     expr3 + expr3.T - np.diag(expr3.diagonal())

        expr3 = cp.atoms.affine.wraps.psd_wrap(expr3)
        expr4 = alpha1 + (r11 - r1)
        q = y_tr

        objective = cp.Minimize((expr1) * cp.quad_form(expr2, expr3) + q.T @ expr4)
        constraints = [0 <= alpha1, alpha1 <= C2 * e / N, r1 + r11 <= 1 - lambda1, 0 <= sum(alpha1),
                       sum(alpha1) <= C2 * nu1, 0 <= r1, 0 <= r11]
        prob = cp.Problem(objective, constraints)
        # results = prob.solve(solver=cp.GUROBI, verbose=True)
        results = prob.solve(cp.GUROBI)

        #obj_val = []
        #obj_val.append(prob.objective.value)

        # 求解TWDWPSVR-2对偶问题

        # 指明参数有助于调节参数以达到最优
        C3 = self.C3
        C4 = self.C4
        nu2 = self.nu2
        lambda2 = self.lambda2

        e = np.ones((N, 1))

        r2 = cp.Variable((N, 1))
        r21 = cp.Variable((N, 1))
        alpha2 = cp.Variable((N, 1))

        expr11 = 1 / (2 * lambda2)
        expr22 = lambda2 * y_tr + (alpha2 + (r2 - r21))
        # expr2 = (lambda1*y-alpha1+e*(r-r1)).T
        expr33 = S_tr @ np.linalg.inv(S_tr.T @ S_tr + C1 * np.identity(l, dtype=int)) @ S_tr.T+0.001*np.identity(N, dtype=int)
        expr33 = np.triu(expr33)
        expr33 =     expr33 + expr33.T - np.diag(expr33.diagonal())
        expr33 = cp.atoms.affine.wraps.psd_wrap(expr33)
        expr44 = alpha2 + (r2 - r21)
        q = -y_tr

        objective = cp.Minimize((expr11) * cp.quad_form(expr22, expr33) + q.T @ expr44)
        # 定义约束条件
        constraints = [0 <= alpha2,
                       alpha2 <= C4 * e / N,
                       r2 + r21 <= 1 - lambda2,
                       0 <= sum(alpha2),
                       sum(alpha2) <= C4 * nu2,
                       0 <= r2, 0 <= r21]

        # 建⽴模型
        prob = cp.Problem(objective, constraints)
        # 模型求解
        # results = prob.solve(solver=cp.GLPK_MI, verbose=True)
        # results = prob.solve(solver=cp.GUROBI, verbose=True)
        results = prob.solve(cp.GUROBI)
        # print('问题的最优值为：{:.0f}'.format(prob.objective.value))
        a=expr2.value
        z1=1/lambda1*np.linalg.inv(S_tr.T@S_tr+C1*np.identity(l, dtype=int))@S_tr.T@a
        b=expr22.value
        z2=1/lambda2*np.linalg.inv(S_tr.T@S_tr+C3*np.identity(l, dtype=int))@S_tr.T@b
        
        self.z1 = z1
        self.z2 = z2
        
        # 计算error
        rmse_tr = 0
        mae_tr = 0
        self.rmse_tr = rmse_tr
        self.mae_tr = mae_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        
    def predict(self, x_te): #仅需要x_tr去训练
        S_te= generateS(x_te)
        z1 = self.z1
        z2 = self.z2
        y_hat1=S_te@z1
        y_hat2=S_te@z2
        y_hat=(y_hat1+y_hat2)/2
        self.y_te_predict = y_hat
        
        rmse_te = 0
        self.rmse_te = rmse_te
        #mae_te = 0
        #self.mae_te = mae_te
        
        self.x_te = x_te
        #self.y_te = y_te
        return y_hat  
    #def random_search2(param_space,x,y, n_iter=500):
    def random_search2(X_train, X_test, y_train, y_test, n_iter=5):
        param_space = {
        'C1': [2**i for i in range(-3, 4)],
        'C2': [2**i for i in range(-8, 9)],
        'nu1': [0.025*i for i in range(1, 20)],
        'lambda1': np.sort(uniform(loc=0.001, scale=0.999).rvs(size=10000)),
        'C3': [2**i for i in range(-3, 4)],
        'C4': [2**i for i in range(-8, 9)],
        'nu2': [0.05*i for i in range(1, 20)],
        'lambda2': np.sort(uniform(loc=0.001, scale=0.999).rvs(size=10000)),
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
                reg=QSSVR(values[0], values[1],values[2], values[3],values[4], values[5],values[6], values[7])
                # print(values[0], values[1],values[2], values[3])
                reg.fit(X_train,y_train)
                y_pred=reg.predict(X_test)   
                score = mean_squared_error(y_test, y_pred, squared=False)+mean_absolute_error( y_test,y_pred)
            
                if score < best_score:
                    best_score = score
                    best_params = values
      
        return best_params
    
