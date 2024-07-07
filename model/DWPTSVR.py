import numpy as np
#import cvxopt
import cvxpy as cp
from sklearn.model_selection import train_test_split 
import random
from sklearn.metrics import r2_score,root_mean_squared_error, mean_absolute_error
from scipy.stats import uniform, randint
import gurobipy 
# 向量的交叉项表示
# 输入的是矩阵N*n维，N是样本数，n是特征数

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


# 返回的也是N*m矩阵，m对应的是升维后列数

# 产生G矩阵
def generateG(x):
    N, n = x.shape
    e = np.ones((N, 1))
    s = lvec(x)
    z = np.hstack((s, x, e))
    eta = lvec(z)
    r = np.hstack((eta, s))
    G = np.hstack((r, x, e))
    return G


class DWPTSVR:

    """
    :parameter
    C1:
    C2:
    nu1:
    lambda1:
    def __init__(
        self,
        *,
        C1=1.0,
        C2=1.0,
        nu1=0.5,
        lambda1=0.5,

       
    ):

        super().__init__(
            C1=C1,
            C2=C2,
            nu1=nu1,
            lambda1=lambda1,
            
        )
    """
    
    def __init__(self, C1, C2, nu1,lambda1,C3, C4, nu2, lambda2):
        #C1= C3 =16
        #nu1 =nu2 =0.2
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
        self.x = None
        self.y = None
        self.y_tr_predict = None
        self.x_te = None
        self.y_te = None
        self.y_te_predict = None

    def fit(self, x_tr, y_tr):#应该可以用输入x替代G
        C1 = self.C1
        C2 = self.C2
        nu1 = self.nu1
        lambda1 = self.lambda1
        
        G_tr=generateG(x_tr)

        N, l = G_tr.shape
        
        e = np.ones((N, 1))
        alpha1 = cp.Variable((N, 1))
        r1 = cp.Variable((N, 1))
        r11 = cp.Variable((N, 1))
        expr1 = 1 / (2 * lambda1)
        expr2 = lambda1 * y_tr - (alpha1 + (r11 - r1))
        expr3 = G_tr @ np.linalg.inv(G_tr.T @ G_tr + C1 * np.identity(l, dtype=int)) @ G_tr.T
        #eig = np.linalg.eigvals(expr3)
        
        expr3=G_tr @ np.linalg.inv(G_tr.T @ G_tr + C1 * np.identity(l, dtype=int)) @ G_tr.T+0.001*np.identity(N, dtype=int)
        
        expr3 = np.triu(expr3)
        expr3 =     expr3 + expr3.T - np.diag(expr3.diagonal())
        
        expr3 = cp.atoms.affine.wraps.psd_wrap(expr3)
        
        expr4 = alpha1 + (r11 - r1)
        q = y_tr

        #objective = cp.Minimize(q.T @ expr4)
        objective = cp.Minimize((expr1) * cp.quad_form(expr2, expr3) + q.T @ expr4)
        constraints = [0 <= alpha1, alpha1 <= C2 * e / N, r1 + r11 <= 1 - lambda1,
                       sum(alpha1) <= C2 * nu1, 0 <= r1, 0 <= r11]#0 <= sum(alpha1)是冗余约束
        prob = cp.Problem(objective, constraints)

        results = prob.solve(solver=cp.GUROBI)
        #results = prob.solve(solver='COPT')

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
        expr33 = G_tr @ np.linalg.inv(G_tr.T @ G_tr + C3 * np.identity(l, dtype=int)) @ G_tr.T
        
        #eig = np.linalg.eigvals(expr33)
        expr33=G_tr @ np.linalg.inv(G_tr.T @ G_tr + C3 * np.identity(l, dtype=int)) @ G_tr.T+0.001*np.identity(N, dtype=int)
        
        expr33 = np.triu(expr33)
        expr33 =     expr33 + expr33.T - np.diag(expr33.diagonal())
        
        expr33 = cp.atoms.affine.wraps.psd_wrap(expr33)
        
        expr44 = alpha2 + (r2 - r21)
        q = -y_tr

        #objective = cp.Minimize(  q.T @ expr44)
        objective = cp.Minimize((expr11) * cp.quad_form(expr22, expr33) + q.T @ expr44)
        # 定义约束条件
        constraints = [0 <= alpha2,
                       alpha2 <= C4 * e / N,
                       r2 + r21 <= 1 - lambda2,
                       sum(alpha2) <= C4 * nu2,
                       0 <= r2, 0 <= r21]

        # 建⽴模型
        prob = cp.Problem(objective, constraints)
        # 模型求解
        # results = prob.solve(solver=cp.GLPK_MI, verbose=True)
        results = prob.solve(solver=cp.GUROBI)
        #results = prob.solve(solver='COPT')
        # print('问题的最优值为：{:.0f}'.format(prob.objective.value))
        a=expr2.value
        u1=1/lambda1*np.linalg.inv(G_tr.T@G_tr+C1*np.identity(l, dtype=int))@G_tr.T@a
        b=expr22.value
        u2=1/lambda2*np.linalg.inv(G_tr.T@G_tr+C3*np.identity(l, dtype=int))@G_tr.T@b
        
        self.u1 = u1
        self.u2 = u2

        y1=G_tr@u1
        y2=G_tr@u2
        
        # 计算error
        rmse_tr = 0
        mae_tr = 0
        self.rmse_tr = rmse_tr
        self.mae_tr = mae_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        return y1,y2
        
    def predict(self, x_te): #仅需要x_tr去训练
        G_te= generateG(x_te)
        u1 = self.u1
        u2 = self.u2
        y_hat1=G_te@u1
        #print(y_hat1)
        y_hat2=G_te@u2
        #print(y_hat2)
        y_hat=(y_hat1+y_hat2)/2
        self.y_te_predict = y_hat
        
        rmse_te = 0
        self.rmse_te = rmse_te
        #mae_te = 0
        #self.mae_te = mae_te
        
        self.x_te = x_te
        #self.y_te = y_te
        return y_hat


    #def random_search1(param_space,x,y, n_iter=500):
    #考虑增加迭代次数，会加速
    def random_search1(x,y, n_iter=10): #n_iter=5
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=41)
        param = {
        'C1': [2**i for i in range(-3, 4)],
        'C2': [2**i for i in range(-8, 9)],
        'nu1': [0.025*i for i in range(1, 20)],
        'C3': [2**i for i in range(-3, 4)],
        'C4': [2**i for i in range(-8, 9)],
        'nu2': [0.05*i for i in range(1, 20)],
         }

        param_space = {
        'lambda1': [0.1*i for i in range(1, 10)],
        'lambda2': [0.1*i for i in range(1, 10)],
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
            para = {}
            for key, value in param.items():
                params[key] = random.choice(value)
            val = list(params.values())   
            #values=[1,1,0.5,0.5]
            for i in range(2**n):
                for key, value in dict[f"dict{i}"].items():
                    para[key] = random.choice(value)
                #这里的random要是有规则的random，以便更好的找到表现好的解
                                      
                values = list(para.values())
                reg=DWPTSVR(val[0], val[1],val[2], values[0],val[3], val[4],val[5], values[1])

                reg.fit(X_train,y_train)
                y_pred=reg.predict(X_test)   
                score = root_mean_squared_error(y_test, y_pred)+mean_absolute_error( y_test,y_pred)
            
                if score < best_score:
                    best_score = score
                    best_params = val + values
        
    
        
        return best_params


    
    



