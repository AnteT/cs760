import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

class LagrangePoly:
    def __init__(self, X, Y):
        self.n = len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)
    def basis(self, x, j):
        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.Y[j]
    def interpolate(self, x):
        b = [self.basis(x, j) for j in range(self.n)]
        return np.sum(b, axis=0)

def plot_lagrange_polynomial_and_all_basis(x=None,y=None):
    """placeholder values for x and y until I know what the hell is going on here"""
    lp = LagrangePoly(x, y)
    xx = np.linspace(0,10,100)
    legend_labels=[]
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        legend_labels.append(fr"$\mathit{{x_{{{i}}},y_{{{i}}}}}$")
        plt.plot(xx, lp.basis(xx, i))
        legend_labels.append(fr"$\mathit{{l_{i}}}$")
    plt.xlim((0,10))
    plt.ylim((-10,10))
    plt.xlabel(r"${x}$ between ${[0,10]}$")
    plt.ylabel(r"$\mathit{sin(x)}$")
    plt.plot(xx, lp.interpolate(xx), linestyle=':')
    legend_labels.append(r"$\mathit{l_x}$")
    ### regular legend ###
    plt.legend(legend_labels,loc='upper right')
    ### moving the legend ###
    plt.title(fr"Langrange Interpolation through {len(x)} basis in $\mathit{{f(x)=sin(x)}}$" + '\n')
    plt.savefig("hw2-lagrange-plot-recent.png")
    plt.show()

def plot_lagrange_polynomial(x=None,y=None):
    """placeholder values for x and y until I know what the hell is going on here"""
    lp = LagrangePoly(x, y)
    xx = np.linspace(0,10,100)
    legend_labels=[]
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        legend_labels.append(fr"$\mathit{{x_{{{i}}},y_{{{i}}}}}$")
        plt.plot(xx, lp.basis(xx, i))
        legend_labels.append(fr"$\mathit{{l_{i}}}$")
    plt.xlim((0,10))
    plt.ylim((-10,10))
    plt.xlabel(r"${x}$ between ${[0,10]}$")
    plt.ylabel(r"$\mathit{sin(x)}$")
    plt.plot(xx, lp.interpolate(xx), linestyle=':')
    legend_labels.append(r"$\mathit{l_x}$")
    ### regular legend ###
    plt.legend(legend_labels,loc='upper right')
    ### moving the legend ###
    plt.title(r"Langrange Interpolation through 5 basis in $\mathit{{f(x)=sin(x)}}$" + '\n')
    plt.savefig("hw2-lagrange-plot-recent.png")
    plt.show()

def get_uniform_distribution(a,b,size=100):
    samples = np.random.uniform(a,b,size)
    return samples

def plot_sample(df:pd.DataFrame):
    x_val = df['x']
    y_val = df['y']
    plt.scatter(x_val,y_val)
    plt.xlim(left=0,right=10)
    plt.show()

def generate_training_sample_dataframe(interval:list,size:int) -> pd.DataFrame:
    start,stop = interval
    n_samples = np.random.uniform(start,stop,size)
    df = pd.DataFrame(data=n_samples,columns=['x'])
    df['y'] = np.sin(df['x'])
    return df

def get_lagrange_data(filename='lg_train.csv'):
    """use lg_train.csv for training set and lg_validate.csv for test set"""
    df = pd.read_csv(filename, index_col=0)
    return df

def generate_langrange_model_from_dataframe(df:pd.DataFrame):
    return lagrange(df['x'],df['y'])    

def find_and_return_MSE_and_MAE_loss(model,df_test:pd.DataFrame, verbose=False) -> dict:
    total = len(df_test.index)
    sum_squared_error = 0
    sum_absolute_error = 0
    for row in df_test.itertuples():
        x, y = row.x, row.y
        model_y = model(x)
        squared_error = (model_y-y)**2
        absolute_error = np.abs(model_y-y)
        if verbose:
            print(f'model predict: {model_y}, truth: {y}, | MSE: {squared_error}, MAE: {absolute_error}')
        sum_squared_error += squared_error
        sum_absolute_error += absolute_error
    MSE = sum_squared_error/total
    MAE = sum_absolute_error/total
    return {'MSE': MSE,'MAE':MAE}

def add_gaussian_noise_to_ndarray(data, mean=0, std=1):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def add_gaussian_noise_to_dataframe(df:pd.DataFrame, mean=0, std=1, target_column:str='y') -> pd.DataFrame:
    data = df[target_column].to_numpy(dtype=np.float64)
    noise = np.random.normal(mean, std, data.shape)
    df[target_column] = pd.Series(data + noise)
    return df

######################## run main function ########################
retrain = True
plot = True
include_MSE = False
TRAINING_INTERVAL = [0,10]
TRAINING_SET_SIZE = 100
GAUSSIAN_NOISE_STD = 1

if retrain:
    df_training = generate_training_sample_dataframe(TRAINING_INTERVAL,TRAINING_SET_SIZE)
    df_training = add_gaussian_noise_to_dataframe(df_training, mean=0, std=GAUSSIAN_NOISE_STD,target_column='y')
    lg_polynomial = generate_langrange_model_from_dataframe(df_training) # returns a formula you can call like f(n)
    training_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial,df_training) # returns validation set across the same distribution as model
    df_validation = generate_training_sample_dataframe(TRAINING_INTERVAL,100)
    test_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial, df_validation)
    if not include_MSE:
        training_error_dict.pop('MSE')
        test_error_dict.pop('MSE')
    train_error, test_error = training_error_dict['MAE'], test_error_dict['MAE']
    print(f'trained on {TRAINING_SET_SIZE}\ngaussian noise std: {GAUSSIAN_NOISE_STD}\ntraining error: {train_error}\ntest error: {test_error}')

### plot lagrange poly ###
if plot:
    x, y = df_training['x'], df_training['y']
    plot_lagrange_polynomial(x,y) # first 5 only
    # plot_lagrange_polynomial_and_all_basis(x,y) # all basis polynomials
