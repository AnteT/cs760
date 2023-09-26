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

def plot_sample(df:pd.DataFrame, title=None):
    x_val = df['x']
    y_val = df['y']
    plt.scatter(x_val,y_val)
    # plt.xlim(left=0,right=10)
    plt.title(title)
    plt.show()

def generate_training_sample_dataframe(interval:list,size:int, add_gaussian_noise=True, mean=0, std=1.0, variant='pre') -> pd.DataFrame:
    """variant types: `pre` adds noise to x before generating y, `post` adds noise to x after generating y"""
    start,stop = interval
    x_sample = np.random.uniform(start,stop,size)
    if not add_gaussian_noise:
        df = pd.DataFrame(data=x_sample,columns=['x'])
        df['y'] = np.sin(df['x'])
        return df
    
    noise = np.random.normal(mean, std, len(x_sample))
    if variant == 'pre':
        x_sample = x_sample + noise
        df = pd.DataFrame(data=x_sample,columns=['x'])
        df['y'] = np.sin(df['x'])
    elif variant == 'post':
        df = pd.DataFrame(data=x_sample,columns=['x'])
        df['y'] = np.sin(df['x'])    
        data = df['x'].to_numpy(dtype=np.float64)
        noise = np.random.normal(mean, std, data.shape)
        df['x'] = pd.Series(data + noise)
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


def train_and_validate_with_optional_noise_vary_by_training_size(TRAINING_SET_SIZE_BATCHES = (5,10,15,20,25,50,75,100), ADD_GAUSSIAN_NOISE=False):
    TRAINING_INTERVAL = [0,10]
    RESULTS_HEADERS = ['T_size', 'train_error', 'test_error','formula']
    RESULTS_DATA = []
    SHOW_FORMULA = False
    SHOW_PLOT = False

    TRAINING_SET_SIZE_BATCHES = (5,10,15,20,25,50,75,100) # partial
    df_validation = generate_training_sample_dataframe(TRAINING_INTERVAL,100,add_gaussian_noise=False)

    for training_size in TRAINING_SET_SIZE_BATCHES:
        TRAINING_SET_SIZE = training_size
        df_training = generate_training_sample_dataframe(TRAINING_INTERVAL,TRAINING_SET_SIZE,add_gaussian_noise=ADD_GAUSSIAN_NOISE,std=25,variant='pre') # pre is the right approach
        if SHOW_PLOT:
            plot_sample(df_training, title=f"Data for training size: {TRAINING_SET_SIZE}")
        lg_polynomial = generate_langrange_model_from_dataframe(df_training) # returns a formula you can call like f(n)
        training_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial,df_training) # returns validation set across the same distribution as model
        test_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial, df_validation)
        training_error_dict.pop('MSE')
        test_error_dict.pop('MSE')
        lg_formula = lg_polynomial if SHOW_FORMULA else None
        dyn_lg_formula = f'{lg_formula}' if SHOW_FORMULA else '' + '\n' if SHOW_FORMULA else ''
        size_var, train_error, test_error = TRAINING_SET_SIZE, training_error_dict['MAE'], test_error_dict['MAE']
        RESULTS_DATA.append([size_var, train_error, test_error, lg_formula])
        print(f"""trained on {TRAINING_SET_SIZE}\ngaussian noise added: {ADD_GAUSSIAN_NOISE}\ntraining error: {train_error}\ntest error: {test_error}{dyn_lg_formula}""")

    print(f'\ntraining by varying sample size validated against 100 test samples, results across size:\n')
    df_summary = pd.DataFrame(data=RESULTS_DATA,columns=RESULTS_HEADERS)
    print(df_summary)

def train_and_validate_with_gaussian_noise_vary_by_std(GAUSS_NOISE_STD_BATCHES:tuple = (0.1,0.5,1.0,1.5,2.0,3.0,10.0,100.0)):
    TRAINING_INTERVAL = [0,10]
    TRAINING_SET_SIZE = 100
    RESULTS_HEADERS = ['std', 'train_error', 'test_error','formula']
    RESULTS_DATA = []
    SHOW_FORMULA = False
    SHOW_PLOT = False

    # GAUSS_NOISE_STD_BATCHES = (0.1,0.2,0.3,0.4,0.5,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0,3.5,4.0,5.0,10.0,25.0,50.0,100.0) # full
    GAUSS_NOISE_STD_BATCHES = (0.1,0.5,1.0,1.5,2.0,3.0,10.0,100.0) # partial
    TRAINING_SET_SIZE_BATCHES = (5,10,15,20,25,50,75,100) # partial

    df_validation = generate_training_sample_dataframe(TRAINING_INTERVAL,100,add_gaussian_noise=False)

    for std in GAUSS_NOISE_STD_BATCHES:
        df_training = generate_training_sample_dataframe(TRAINING_INTERVAL,TRAINING_SET_SIZE,add_gaussian_noise=True,std=std,variant='pre') # pre is the right approach
        if SHOW_PLOT:
            plot_sample(df_training, title=f"Data for STD: {std}")
        lg_polynomial = generate_langrange_model_from_dataframe(df_training) # returns a formula you can call like f(n)
        training_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial,df_training) # returns validation set across the same distribution as model
        test_error_dict = find_and_return_MSE_and_MAE_loss(lg_polynomial, df_validation)
        training_error_dict.pop('MSE')
        test_error_dict.pop('MSE')
        lg_formula = lg_polynomial if SHOW_FORMULA else None
        dyn_lg_formula = f'{lg_formula}' if SHOW_FORMULA else '' + '\n' if SHOW_FORMULA else ''
        std_var, train_error, test_error = std, training_error_dict['MAE'], test_error_dict['MAE']
        RESULTS_DATA.append([std_var, train_error, test_error, lg_formula])
        print(f"""trained on {TRAINING_SET_SIZE}\ngaussian noise std: {std}\ntraining error: {train_error}\ntest error: {test_error}{dyn_lg_formula}""")

    print(f'\ntraining with {TRAINING_SET_SIZE} samples validated against 100 test samples, results across std:\n')
    df_summary = pd.DataFrame(data=RESULTS_DATA,columns=RESULTS_HEADERS)
    print(df_summary)

######################## run main function ########################
### train and test varying gaussian noise ###
train_and_validate_with_gaussian_noise_vary_by_std(GAUSS_NOISE_STD_BATCHES = (0.1,0.5,1.0,1.5,2.0,3.0,10.0,100.0))

### train and test varying test size ###
train_and_validate_with_optional_noise_vary_by_training_size(TRAINING_SET_SIZE_BATCHES = (5,10,15,20,25,30,40,50,75,100), ADD_GAUSSIAN_NOISE=True)
