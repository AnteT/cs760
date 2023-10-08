### function dependencies for homework 3 ###
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve
from sklearn.neighbors import KNeighborsClassifier as KNC # used for validating custom approaches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_roc(savefile:str=None,color='#afd6d2'):
    fontfamily = {'fontname':'GE Inspira Sans'}
    q5_data = ((1, .95),(1, .85),(0, .8),(1, .7),(1, .55),(0, .45),(1, .4),(1, .3),(0, .2),(0, .1))
    q5_conf = [x[1] for x in q5_data]
    y = [x[0] for x in q5_data]
    fpr = [0.0]
    tpr = [0.0]
    P = sum(y)
    N = len(y) - P
    for thresh in q5_conf:
        FP, TP = 0, 0
        for i in range(len(q5_conf)):
            if (q5_conf[i] >= thresh):
                if y[i] == 1:
                    TP = TP + 1
                elif y[i] == 0:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))
    plt.scatter(fpr, tpr, c=color)
    plt.plot(fpr, tpr, c=color)
    plt.text(0.020, 0.28, r'$\mathit{P_{ threshold}}$', c=color,fontsize=10)
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e')
    d_xlabel = 'False positive rate'
    dy_ylabel = 'True positive rate'
    plt.xlabel(d_xlabel, fontsize=11, **fontfamily) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=11, **fontfamily) # set y-axis label
    plt.title("Spam filter ROC curve\n", **fontfamily,fontsize=14)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

def fetch_dataset(filename:str=None, return_as_cooridinates:bool=False) -> tuple:
    """made for D2z.txt and emails.csv"""
    filetype = filename.split('.')[-1]
    if filetype == 'txt':
        df = pd.read_csv(filename, sep=' ',names=['x1','x2','y'])
        starting_col = 1
    else:
        df = pd.read_csv(filename)
        starting_col = 2
    return (df, tuple(np.array(x[starting_col:]) for x in df.itertuples())) if return_as_cooridinates else (df, None)

def D2z_1NN(filename:str=None, plot:bool=False, savefile:str=None):
    df_train, train_points = fetch_dataset(filename,return_as_cooridinates=True)
    test_points =[np.array((np.around(x1,2), np.around(x2,2))) for x1 in np.arange(-2, 2, .1) for x2 in np.arange(-2, 2, .1)]
    df_preds = pd.DataFrame(columns=['x1','x2','y'])
    num_total = len(test_points)
    num_completed = 0
    for test_p in test_points:
        nearest_neighbor = (np.inf, None)
        for train_p in train_points:
            pred = train_p[-1]
            dist = np.linalg.norm(test_p - train_p[:-1])
            if dist < nearest_neighbor[0]:
                nearest_neighbor = (dist, pred)
        df_preds.loc[len(df_preds.index)] = [test_p[0], test_p[1], nearest_neighbor[-1]]
        num_completed += 1
        print(f'{num_completed}/{num_total} completed')
    if not plot:
        return
    plum_lb_cmap = LinearSegmentedColormap.from_list("", ['#c8a4c6','#afd6d2'])
    plt.scatter(df_preds.x1, df_preds.x2, c=df_preds.y,s=9,cmap=plum_lb_cmap) # YlOrRd
    plt.scatter(df_train.x1, df_train.x2, c=df_train.y, marker="o",cmap=plum_lb_cmap)
    
    ### styling portion ###
    fontfamily = {'fontname':'GE Inspira Sans'}
    d_xlabel = 'X1'
    dy_ylabel = 'X2'
    plt.xlabel(d_xlabel, fontsize=11, **fontfamily) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=11, **fontfamily) # set y-axis label
    plt.title("D2z 1NN Plot\n", **fontfamily,fontsize=14)
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

### 5 fold emails.csv cross validation training ###
def nearest_neighbor_by_k(top_k:int=3, fold:int=5, return_df:bool=False):
    print('beginning...')
    df = fetch_dataset(filename='HW3/data/emails.csv')[0]
    df_cols = df.columns.tolist()
    X = df[df_cols[1:-1]].values
    y = df[df_cols[-1]].values
    all_results = []
    if fold is not None:
        idx_split = np.split(np.arange(len(df.index)), fold)
        for i in range(fold):
            print(f'beginning fold {i+1}...')
            test_idx = idx_split[i]
            train_idx = list(set(np.arange(len(df.index))) - set(idx_split[i]))
            X_test, y_true = X[test_idx], y[test_idx]
            X_train, y_train = X[train_idx], y[train_idx]
            y_pred = find_all_k_nearest(X_train, y_train, X_test, k=top_k)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            print(f'fold {i+1} finished')
    else:
        X_test, y_true = X[4000:], y[4000:]
        X_train, y_train = X[:4000], y[:4000]
        y_pred = find_all_k_nearest(X_train, y_train, X_test, k=top_k)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)        
        all_results.append([i+1,accuracy,precision,recall])
    df_results = pd.DataFrame(data=all_results,columns=['fold','accuracy','precision','recall'])
    print(df_results)
    if return_df:
        return (df_results, y)

def find_all_k_nearest(X_train, y_train, X_predict, k):
    y_predict = [find_nearest_k(X_train, y_train, x, k) for x in X_predict]
    return np.array(y_predict)

def find_nearest_k(X_train, y_train, x, k):
    distances = np.linalg.norm(x - X_train, axis = 1)
    nearest = np.argpartition(distances,k)[:k]
    topK_y = y_train[nearest]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]

class LogitReg:
    """class to persist weights and biases across iterations, implement logistic regression from scratch"""
    def __init__(self, learning_rate=0.01, num_iterations=1_000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None # assume we're not initializing random weights for this homework
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features,dtype=np.float64)
        self.bias = 0
        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sig(z)
            delta_weights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            delta_bias = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * delta_weights
            self.bias -= self.learning_rate * delta_bias
            
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sig(z)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def sig(self, x):
        return np.array([self.sig_overflow_hack(value) for value in x])

    def sig_overflow_hack(self, x):
        """some sort of float overflow hack I found online, works great"""
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)    

def begin_training(lr:float=0.01, num_iter:int=1_000, k_fold:int = 5, randomize:bool=False):
    df = pd.read_csv("HW3/data/emails.csv")
    if randomize:
        df = df.sample(frac=1.0)
    X_email = df.drop(columns=['Email No.', 'Prediction'])
    y_email = df['Prediction']    
    lr_model = LogitReg(learning_rate=lr, num_iterations=num_iter)
    all_idx = np.arange(5000)
    idxs = np.split(all_idx, k_fold)
    X = X_email.values
    y = y_email.values

    for i in range(k_fold):
        test_idx = idxs[i]
        train_idx = list(set(all_idx) - set(idxs[i]))
        X_test, y_true = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = np.around(accuracy_score(y_true, y_pred),6)
        precision = np.around(precision_score(y_true, y_pred),6)
        recall = np.around(recall_score(y_true, y_pred),6)
        print(f"""k_fold: {i+1}\taccuracy: {accuracy}\tprecision: {precision}\trecall: {recall}""")
        return {'fold':i+1,'accuracy':accuracy,'precision':precision,'recall':recall} # return results as dict

def five_fold_CV_across_varying_K(k_neighbors:tuple=(1,3,5,7,10), savefile:str=None):
    df = pd.read_csv('HW3/data/emails.csv')
    y = df['Prediction']
    df = df.drop(columns=['Email No.', 'Prediction'])
    k_n_results = []
    for k_n in k_neighbors:
        df = nearest_neighbor_by_k(top_k=k_n)
        k_n_results.append([k_n,np.around(df['accuracy'].mean(),4)])
    fontfamily = {'fontname':'GE Inspira Sans'}
    plt.xlabel('K', fontsize=11, **fontfamily) # set x-axis label
    plt.ylabel('Average Accuracy', fontsize=11, **fontfamily) # set y-axis label
    plt.title("KNN 5-fold CV across varying K\n", **fontfamily,fontsize=14)
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    plum_c = '#c8a4c6'
    lb_c = '#afd6d2'
    plt.plot([x[0] for x in k_n_results], [x[1] for x in k_n_results], color=lb_c)
    plt.scatter([x[0] for x in k_n_results], [x[1] for x in k_n_results], color=lb_c)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()   
    df_results = pd.DataFrame(data=k_n_results,columns=('K','accuracy'))
    print(df_results)


def five_fold_CV_across_varying_K_sklearn_check(k_neighbors:tuple=(1,3,5,7,10), savefile:str=None):
    """sklearn version to confirm results"""
    df = pd.read_csv('HW3/data/emails.csv')
    y = df['Prediction']
    df = df.drop(columns=['Email No.', 'Prediction'])
    k_n_results = []
    for k_n in k_neighbors:
        neigh = KNC(n_neighbors=k_n)
        cv = KFold(n_splits=5, shuffle=False)
        scores = cross_val_score(neigh, df, y, scoring='accuracy', cv=cv)
        k_n_results.append([k_n,np.around(np.mean(scores),4)])
    fontfamily = {'fontname':'GE Inspira Sans'}
    plt.xlabel('K', fontsize=11, **fontfamily) # set x-axis label
    plt.ylabel('Average Accuracy', fontsize=11, **fontfamily) # set y-axis label
    plt.title("KNN 5-fold CV across varying K\n", **fontfamily,fontsize=14)
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    plum_c = '#c8a4c6'
    lb_c = '#afd6d2'
    plt.plot([x[0] for x in k_n_results], [x[1] for x in k_n_results], color=lb_c)
    plt.scatter([x[0] for x in k_n_results], [x[1] for x in k_n_results], color=lb_c)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()   
    df_results = pd.DataFrame(data=k_n_results,columns=('K','accuracy'))
    print(df_results)

def knn_vs_logistic_regression(savefile:str=None):
    df = pd.read_csv('HW3/data/emails.csv')
    y = df['Prediction']
    df = df.drop(columns=['Email No.', 'Prediction'])
    x_train= df[:4000]
    x_test = df[4000:5000]
    y_train = y[:4000]
    y_test = y[4000:5000]
    neighbors, y_neighbors = nearest_neighbor_by_k(top_k=5,fold=None,return_df=True)
    print(f'finished knn, returning np array for intermediate check:\n{y_neighbors}')
    logistic = LogitReg()
    logistic.fit(x_train, y_train)
    pred_y_knn = np.array(neighbors.y)[:,-1]
    pred_y_lr = np.array(logistic.predict(x_test))[:,1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, pred_y_knn)
    knn_auc = auc(knn_fpr, knn_tpr)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, pred_y_lr)
    lr_auc = auc(lr_fpr, lr_tpr)

    ### plot styling ###
    font_dict = {'fontname':'GE Inspira Sans','fontsize':11}
    color_dict = {'plum':'#c8a4c6', 'lb':'#afd6d2'}

    plt.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
    plt.title('Logistic Regression vs K-Nearest Neighbors ROC AUC\n',**font_dict)
    knn_label = f"K-Nearest Neighbors AUC: {'{:.4f}'.format(knn_auc)}"
    lr_label = f"Logistic Regression AUC: {'{:.4f}'.format(lr_auc)}"
    plt.plot(lr_fpr, lr_tpr, label=lr_label, color=color_dict['lb'],linewidth=2)
    plt.plot(knn_fpr, knn_tpr, label=knn_label, color=color_dict['plum'],linewidth=2) # increasing line width to see better
    legend = plt.legend(facecolor='#fafbfd')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor('#2b2d2e')
    for text in legend.get_texts():
        text.set_color('#fafbfd')
        text.set_fontfamily('GE Inspira Sans')
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate (positive label: 1)',**font_dict)
    plt.xlabel('False Positive Rate (positive label: 1)',**font_dict)
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e')
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

def knn_vs_logistic_regression_sklearn_check(savefile:str=None):
    """sklearn version to confirm results"""
    df = pd.read_csv('HW3/data/emails.csv')
    y = df['Prediction']
    df = df.drop(columns=['Email No.', 'Prediction'])
    x_train= df[:4000]
    x_test = df[4000:5000]
    y_train = y[:4000]
    y_test = y[4000:5000]
    neighbors = KNC(n_neighbors=5)
    neighbors.fit(x_train, y_train)
    logistic = LogitReg()
    logistic.fit(x_train, y_train)
    pred_y_knn = np.array(neighbors.predict_proba(x_test))[:,1]
    pred_y_lr = np.array(logistic.predict_proba(x_test))[:,1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, pred_y_knn)
    knn_auc = auc(knn_fpr, knn_tpr)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, pred_y_lr)
    lr_auc = auc(lr_fpr, lr_tpr)

    ### plot styling ###
    font_dict = {'fontname':'GE Inspira Sans','fontsize':11}
    color_dict = {'plum':'#c8a4c6', 'lb':'#afd6d2'}

    plt.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
    plt.title('Logistic Regression vs K-Nearest Neighbors ROC AUC\n',**font_dict)
    knn_label = f"K-Nearest Neighbors AUC: {'{:.4f}'.format(knn_auc)}"
    lr_label = f"Logistic Regression AUC: {'{:.4f}'.format(lr_auc)}"
    plt.plot(lr_fpr, lr_tpr, label=lr_label, color=color_dict['lb'],linewidth=2)
    plt.plot(knn_fpr, knn_tpr, label=knn_label, color=color_dict['plum'],linewidth=2) # increasing line width to see better
    legend = plt.legend(facecolor='#fafbfd')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor('#2b2d2e')
    for text in legend.get_texts():
        text.set_color('#fafbfd')
        text.set_fontfamily('GE Inspira Sans')
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate (positive label: 1)',**font_dict)
    plt.xlabel('False Positive Rate (positive label: 1)',**font_dict)
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e')
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

### ============================================= logistic regression functional approach deps ============================================= ###
def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

def helper_return_correct_pred_or_not(row):
    y_fact, y_pred = row.Prediction, row.pred
    if y_fact == y_pred:
        return 'correct'
    return 'incorrect'

def calculate_metrics(df:pd.DataFrame):
    num_total = len(df.index)
    TP = len(df[((df['Prediction']==1) & (df['pred']==1))].index)
    FP = len(df[((df['Prediction']==0) & (df['pred']==1))].index)
    FN = len(df[((df['Prediction']==1) & (df['pred']==0))].index)
    num_correct = len(df[df['result']=='correct'].index)
    num_incorrect = len(df[df['result']=='incorrect'].index)
    accuracy = num_correct/num_total
    precision = TP/(TP + FP)
    recall = TP / (TP + FN)
    accuracy = '{:.2%}'.format(accuracy)
    precision = '{:.2%}'.format(precision)
    recall = '{:.2%}'.format(recall)
    return (num_total,num_correct,num_incorrect,accuracy,precision,recall)

def run_training_cycle(lr:float=0.01, num_iter:int=1_000, test_train_split:float=0.2, randomize:bool=False):
    df = pd.read_csv('HW3/data/emails.csv')
    total = len(df.index)
    if randomize:
        df = df.sample(frac=1.0)
        df.reset_index(drop=True,inplace=True)
    df = df.drop(['Email No.'],axis=1)

    train_size = total - round(total * test_train_split)
    df_train = df[:train_size]
    df_test = df[train_size:]

    print(f"beginning {num_iter} num_iter with learning rate: {lr} using train test split: {train_size}, {total-train_size}")
    feature_cols = df_train.columns.to_list()
    feature_cols = feature_cols[:-1]
    X = df_train[feature_cols]
    y = df_train['Prediction']
    X_test = df_test[feature_cols]
    y_test = df_test['Prediction']
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    theta = np.zeros(X.shape[1])
    for i in range(num_iter):
        h = sigmoid(X, theta)
        if i != 0 and i % 100 == 0:
            print(f'{i}/{num_iter} iterations...')
        gradient = gradient_descent(X, h, y)
        theta = update_weight_loss(theta, lr, gradient)
    result = sigmoid(X, theta)
    training_df = pd.DataFrame(np.around(result, decimals=6),columns=['theta']).join(y)
    training_df['pred'] = training_df['theta'].apply(lambda x : 0 if x < 0.5 else 1)
    training_df['result'] = training_df.apply(helper_return_correct_pred_or_not,axis=1)
    print(training_df)
    metrics_data = []
    num_total,num_correct,num_incorrect,accuracy,precision,recall = calculate_metrics(training_df)
    metrics_data.append(['training',num_total,num_correct,num_incorrect,accuracy,precision,recall])
    test_result = sigmoid(X_test,theta)
    y_test.reset_index(drop=True,inplace=True)
    testing_df = pd.DataFrame(np.around(test_result, decimals=6),columns=['theta']).join(y_test)
    testing_df['pred'] = testing_df['theta'].apply(lambda x : 0 if x < 0.5 else 1)
    testing_df['result'] = testing_df.apply(helper_return_correct_pred_or_not,axis=1)
    print(testing_df)
    num_total,num_correct,num_incorrect,accuracy,precision,recall = calculate_metrics(testing_df)
    metrics_data.append(['test',num_total,num_correct,num_incorrect,accuracy,precision,recall])
    metrics_headers = ['type','total','correct','incorrect','accuracy','precision','recall']
    df_metrics = pd.DataFrame(data=metrics_data,columns=metrics_headers)
    print(df_metrics)

### ============================================= logistic regression functional approach deps ============================================= ###