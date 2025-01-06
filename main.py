from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, pi
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata
from scipy import integrate

def biweight_univariate_density(u):
    if type(u) == float:
        u = np.array([u])
    res = np.where(u**2 <= 1, (15/16)*((1-u**2)**2), 0)
    return res 


def local_dependence_estimator(samples,x,y): #Jones and Koch (2003), this function itself is not used
    kernel = biweight_univariate_density
    n = len(samples)
    sigma1 = np.std(samples.T[0])#variance
    sigma2 = np.std(samples.T[1])#variance
    rho = np.corrcoef(samples.T)[0][1]#correlation 
    term1 = 5/7 #gaussian kernelなら1/2sqrt(pi)
    term2 = 1/7 #gaussian kernel なら1
    common = (2*sqrt(pi)*term1/term2)**(1/3)
    common *= (1-rho**2)**(5/12)
    common /= (1+(rho**2)/2)**(1/6)
    common /= n**(1/6)
    h1 = sigma1 * common
    h2 = sigma2 * common
    print("Bandwidths:",h1,h2)

    def g(i,j,samples_,x0,y0):
        res = []
        for k in range(n):
            Xk = samples_[k][0]
            Yk = samples_[k][1]     
            tmp = (Xk**i)*(Yk**j)*kernel((Xk-x0)/h1)*kernel((Yk-y0)/h2)
            
            tmp /= h1
            tmp /= h2
            res.append(tmp)
        return sum(res)/len(res)
    
    g00 = g(0,0,samples,x,y)
    
    est = g(1,1,samples,x,y)-g(0,1,samples,x,y)*g(1,0,samples,x,y)/g00
    est /= g00
    est /= h1**2
    est /= h2**2
    est /= term2**2
    # print((h1*h2*term2)**2)
    return est,g00,(sigma1,sigma2)

def relative_local_dependence_bilinear_estimator(samples,x,y,band): #Naive
    n = len(samples)
    kernel = biweight_univariate_density
    sigma1 = np.std(samples.T[0])#variance
    sigma2 = np.std(samples.T[1])#variance
    rho = np.corrcoef(samples.T)[0][1]#correlation 
    term1 = 5/7 #gaussian kernelなら1/2sqrt(pi)
    term2 = 1/7 #gaussian kernel なら1
    common = (2*sqrt(pi)*term1/term2)**(1/3)
    common *= (1-rho**2)**(5/12)
    common /= (1+(rho**2)/2)**(1/6)
    common /= n**(1/6)
    h1 = sigma1 * common
    h2 = sigma2 * common
    if band:
        h1, h2 = band
    h1, h2 = h1*2, h2*2
    print("Bandwidths:",h1,h2)

    def g(i,j,samples_,x0,y0):
        res = []
        for k in range(n):
            Xk = samples_[k][0]
            Yk = samples_[k][1]     
            tmp = (Xk**i)*(Yk**j)*kernel((Xk-x0)/h1)*kernel((Yk-y0)/h2)
            
            tmp /= h1
            tmp /= h2
            res.append(tmp)
        return sum(res)/len(res)
    
    g00 = g(0,0,samples,x,y)
    
    est = g(1,1,samples,x,y)-g(0,1,samples,x,y)*g(1,0,samples,x,y)/g00
    est /= g00 ** 2 #g00で1回余分に割っている
    est /= h1**2
    est /= h2**2
    est /= term2**2
    # print((h1*h2*term2)**2)
    return est,g00,(sigma1,sigma2)

### Local Frank fitting (Ours)
def frank_density(theta,u,v):
    res = theta*(1-np.exp(-theta))*np.exp(-theta*(u+v))
    res = res / (1-np.exp(-theta)-(1-np.exp(-theta*u))*(1-np.exp(-theta*v)))**2
    return res

def log_frank_density(theta,u,v):
    return np.log(frank_density(theta,u,v))

def clayton_cdf(alpha, u, v):
    return 1/(u**-alpha + v**-alpha - 1)**(1/alpha)

def K(u):
    if type(u) == float:
        u = np.array([u])
    res = np.where(u**2 <= 1, (15/16)*((1-u**2)**2), 0)
    return res 

def get_rule_of_thumb_bandwidth(samples):
    n = len(samples)
    sigma1 = np.std(samples.T[0])#variance
    sigma2 = np.std(samples.T[1])#variance
    rho = np.corrcoef(samples.T)[0][1]#correlation 
    term1 = 5/7#gaussian kernelなら1/2sqrt(pi)
    term2 = 1/7#gaussian kernel なら1
    common = (2*sqrt(pi)*term1/term2)**(1/3)
    common *= (1-rho**2)**(5/12)
    common /= (1+(rho**2)/2)**(1/6)
    common /= n**(1/6)
    h1 = sigma1 * common 
    h2 = sigma2 * common
    return h1,h2

def objective(x,y,data,theta,band):
    if not band:
        h1, h2 = get_rule_of_thumb_bandwidth(data)
    else:
        h1, h2 = band ###Change if needed
    # print("Bandwidth calcuated by rule of thumb: ", h1,h2)

    def Kh1(t):
        return K(t/h1)/h1

    def Kh2(t):
        return K(t/h2)/h2

    def get_double_integral(x_tmp,y_tmp):
        def f(X,Y):
            return Kh1(X-x_tmp)*Kh2(Y-y_tmp)*frank_density(theta,X,Y)
        return integrate.dblquad(f,0,1,0,1)[0]
    
    tmp_ = 0
    for i in range(len(data)):
        xi,yi = data[i][0],data[i][1]
        tmp_ += Kh1(xi-x)*Kh2(yi-y)*log_frank_density(theta,xi,yi)
        
    tmp_ /= len(data)
    tmp2_ = get_double_integral(x,y)
    # print(tmp_,tmp2_)
    return tmp_ - tmp2_

def argmax_objective(x,y,data,band):
    res = []
    alpha = 3
    # true_value = 2 * alpha ### When ground truth is Frank copula 
    true_value = ((1/clayton_cdf(alpha,x,y))* alpha * (1+2*alpha) / (1+alpha)) ### When ground truth is Clayton copula 
    true_value = true_value/2
    l = np.arange(true_value-4,true_value+4,0.1)
    for k in l: #grid search that maximizes the likelihood
        res.append(objective(x,y,data,theta=k,band=band))
    ind = res.index(max(res))
    return l[ind]

def relative_local_dependence_frank_estimator(x,y,data,band):
    return 2 * argmax_objective(x,y,data,band)  #2θ_hatだから2倍している


def preprocess(data):
    n = len(data)
    R = (rankdata(data.T,axis=1)-0.5)/n
    return R.T

def save(arr, filename, grid=False):
    plt.figure(figsize=(10,10))
    plt.scatter(arr[:,0],arr[:,1])
    if grid:
        for x in arr[:,0].tolist():
            plt.vlines(x, 0, 1, colors="gray", linewidth=1)
        for y in arr[:,1].tolist():
            plt.hlines(y, 0, 1, colors="gray", linewidth=1)
    plt.savefig(f"{filename}.png")

def run(mode,data,save_flag):
    N = 10
    x_grid = np.linspace(0.15,0.95,N)
    y_grid = np.linspace(0.15,0.95,N)
    # x_grid = np.linspace(-2,2,N)
    # y_grid = np.linspace(-2,2,N)

    left_bottom = (0,0)
    right_top = (1,1)
    # left_bottom = (-2,-2)
    # right_top = (2,2)

    grid = N
    X, Y = np.mgrid[left_bottom[0]*grid:right_top[0]*grid, left_bottom[1]*grid:right_top[1]*grid]/grid + 1/(2*N)
    use_mask = False


    start_time = time()
    if mode == "bilinear":    # Local bilinear fitting
        print("LOCAL BILINEAR FITTING...")    
        Z_new,f1,sd = relative_local_dependence_bilinear_estimator(data,X,Y,band=None)
        if use_mask:### Put a MASK where data points are too few
            cutoff = np.quantile(a = f1, q = [0.6])
            masked_new = np.where(f1 < cutoff,0,Z_new)
        else:## or No MASK
            masked_new = Z_new
        
        fig = plt.figure(figsize=(10, 5), facecolor="w")
        ax = fig.add_subplot(121, projection="3d")
        ax.set_zlim(0, 20)
        surf = ax.plot_surface(X, Y, masked_new)
    
    elif mode == "bilinear_ppf":    # Local bilinear fitting
        print("LOCAL BILINEAR FITTING after ppf ...") 
        data_ppf = norm.ppf(data) #marginal transformation Φ^{-1}
        X_ppf = norm.ppf(X)
        Y_ppf = norm.ppf(Y)
        Z_new,f1,sd = relative_local_dependence_bilinear_estimator(data_ppf,X_ppf,Y_ppf,band=None)
        if use_mask:### Put a MASK where data points are too few
            cutoff = np.quantile(a = f1, q = [0.6])
            masked_new = np.where(f1 < cutoff,0,Z_new)
        else:## or No MASK
            masked_new = Z_new
        
        fig = plt.figure(figsize=(10, 5), facecolor="w")
        ax = fig.add_subplot(121, projection="3d")
        ax.set_zlim(0, 20)
        surf = ax.plot_surface(X, Y, masked_new)
        
    elif mode == "frank":    # Local Frank fitting
        print("LOCAL FRANK FITTING...")
        
        Z_new = np.zeros((N,N))
        for i in tqdm(range(N)):
            for j in range(N):
                x,y = x_grid[i],y_grid[j]
                Z_new[i][j] = relative_local_dependence_frank_estimator(x,y,data,band=None) 

        if use_mask:### Put a MASK where data points are too few
            cutoff = np.quantile(a = f1, q = [0.6])
            masked_new = np.where(f1 < cutoff,0,Z_new)
        else:## or No MASK
            masked_new = Z_new

        fig = plt.figure(figsize=(10, 5), facecolor="w")
        ax = fig.add_subplot(121, projection="3d")
        ax.set_zlim(0, 20)
        surf = ax.plot_surface(X, Y, masked_new)
    
    end_time = time()
    print("Total computational time (s) = ", end_time-start_time)

    plt.savefig(f"output_{save_flag}.png")
    np.savetxt(f'Z_{save_flag}.txt', Z_new)

    ### Groud Truth
    GT = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            alpha = 3
            x = X[i][j]
            y = Y[i][j]
            GT[i][j] = ((1/clayton_cdf(alpha,x,y))* alpha * (1+2*alpha) / (1+alpha))
    np.savetxt('GT.txt', GT)


if __name__ == "__main__":
     # Data Sample
    from statsmodels.distributions.copula.api import FrankCopula, ClaytonCopula
    # data = FrankCopula(theta=3, k_dim=2).rvs(nobs=1000)
    data = ClaytonCopula(theta=3, k_dim=2).rvs(nobs=1000)

    save(data,"orig_data_tmp",grid=False)
    rank_data = preprocess(data)
    save(rank_data,"rank_data_tmp",grid=False)

    run(mode="bilinear",data=rank_data,save_flag="bilinear_rankdata")
    run(mode="bilinear_ppf",data=rank_data,save_flag="bilinear_ppf_rankdata")
    run(mode="frank",data=rank_data,save_flag="frank_rankdata")

    M1 = np.loadtxt("Z_bilinear_rankdata.txt")
    M2 = np.loadtxt("Z_bilinear_ppf_rankdata.txt")
    M3 = np.loadtxt("Z_frank_rankdata.txt")
    GT = np.loadtxt("GT.txt")

    mask = np.ones((10,10))
    for i in range(10):
        for j in range(10):
            if abs(i - j) >= 6:
                mask[i][j] = 0

    M1 = M1 * mask
    M2 = M2 * mask
    M3 = M3 * mask
    GT = GT

    datasets = [GT,M1,M2,M3]
    fig, axs = plt.subplots(1, 3)
    from matplotlib import colors
    norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))
    images = []
    for ax, data in zip(axs.flat, datasets):
        images.append(ax.imshow(data, norm=norm))
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
    fig.savefig("2D.png")



