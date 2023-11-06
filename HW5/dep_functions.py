import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # sanity checks to compare scratch implementation
from sklearn.mixture import GaussianMixture # sanity checks to compare scratch implementation
from sklearn.metrics import mean_squared_error as MSE # mean squared error ready made
from sklearn.preprocessing import StandardScaler as SS # for scaling matrix input for pca functions

def get_samples_for_distribution_by_scalar(distribution:str, scalar:float=None, return_class:bool=False, display_properties:bool=False) -> np.ndarray:
    """draws 100 samples from Pa, Pb, Pc after scaling by arg and returns vstacked with class optionally appended to data"""
    match distribution.lower()[-1]:
        case 'a':
            mu, std, label = np.array([-1,-1]), np.array([[2,.5],[.5,1]]), 0
        case 'b':
            mu, std, label = np.array([1,-1]), np.array([[1,-.5],[-.5,2]]), 1
        case 'c':
            mu, std, label = np.array([0,1]), np.array([[1,0],[0,2]]), 2    
        case other:
            return
    sample = np.random.multivariate_normal(mu, (np.dot(scalar,std)), size=100) # draw from normal distribution using properties of sample
    sample = np.c_[sample, np.full(sample.shape[0], label)] # append class label as third column to sample
    if display_properties:
        sample_mu = np.around(sample[:,:2].mean(axis=0),4)
        sample_std = np.around(np.cov(sample[:,:2].T),4)
        print(f'{"-"*80}\nproperties for target distribution: {distribution} (class = {label})\ntarget mu:\n{mu}\ntarget std:\n{std}\nsample mu:\n{sample_mu}\nsample std:\n{sample_std}\n(scalar factor: {scalar})')
    sample = np.around(sample,6)
    if return_class:
        return sample
    return sample[:,:2]

def accuracy(model, x_data):
    predictions = model.predict(x_data)
    values, counts = np.unique(predictions, return_counts=True, axis=0)
    presumed_class = np.argmax(counts, axis=0)
    return counts[presumed_class] / np.sum(counts)

def objective(model, centroids, data):
    predicted_means = centroids[model.predict(data)]
    distances = np.linalg.norm(predicted_means - data, axis=0)
    return np.sum(np.square(distances))

def get_gmm_accuracy(gmm_model, sigma, X_data, y_data):
    pseudo_marker = np.array([[-1,-1],[0,1],[1,-1]]) * sigma # 0, 2, 1
    distribution_markers = np.array([0,2,1])
    derived_markers = gmm_model.predict(pseudo_marker)
    derived_mapping_dict = {derived_markers[i]:distribution_markers[i] for i in range(len(derived_markers))}
    y_predictions = gmm_model.predict(X_data)
    if len(derived_mapping_dict.keys()) < 3:
        derived_mapping_dict = {
            np.argmax(np.bincount(y_predictions[:100])): 0
            ,np.argmax(np.bincount(y_predictions[100:200])): 1
            ,np.argmax(np.bincount(y_predictions[200:])): 2
        }
        if len(derived_mapping_dict.keys()) < 3:
            key_maps = (0,1,2)
            for missing_key in key_maps:
                if missing_key not in derived_mapping_dict.keys():
                    for missing_value in key_maps:
                        if missing_value not in derived_mapping_dict.values():
                            derived_mapping_dict[missing_key] = missing_value
        print(f'created bincount mapping due to missing prediction mappings:\n{derived_mapping_dict}')
    correct, total = 0, 300
    for i in range(len(y_predictions)):
        y_true = y_data[i]
        y_pred = derived_mapping_dict[y_predictions[i]]
        if y_true == y_pred:
            correct += 1
    model_accuracy = round(correct/total,4)
    return model_accuracy

def get_samples_for_distribution_by_scalar(distribution:str, scalar:float=None, return_class:bool=False, display_properties:bool=False) -> np.ndarray:
    """draws 100 samples from Pa, Pb, Pc after scaling by arg and returns vstacked with class optionally appended to data"""
    match distribution.lower()[-1]:
        case 'a':
            mu, std, label = np.array([-1,-1]), np.array([[2,.5],[.5,1]]), 0
        case 'b':
            mu, std, label = np.array([1,-1]), np.array([[1,-.5],[-.5,2]]), 1
        case 'c':
            mu, std, label = np.array([0,1]), np.array([[1,0],[0,2]]), 2    
        case other:
            print(f'Error: unrecognized distribution provided: {distribution}')
            return
    sample = np.random.multivariate_normal(mu, (np.dot(scalar,std)), size=100) # draw from normal distribution using properties of sample
    sample = np.c_[sample, np.full(sample.shape[0], label)] # append class label as third column to sample
    if display_properties:
        sample_mu = np.around(sample[:,:2].mean(axis=0),4)
        sample_std = np.around(np.cov(sample[:,:2].T),4)
        print(f'{"-"*80}\nproperties for target distribution: {distribution} (class = {label})\ntarget mu:\n{mu}\ntarget std:\n{std}\nsample mu:\n{sample_mu}\nsample std:\n{sample_std}\n(scalar factor: {scalar})')
    sample = np.around(sample,6)
    if return_class:
        return sample
    return sample[:,:2]

def get_x_train_y_train_stacked_samples(sigma:float=0.5, sorted:bool=True):
    X = np.zeros(shape=(0,3)) # set to 0,3 if returning class, otherwise 0,2
    for distribution in ('a', 'b', 'c'):
        gaussian_sample = get_samples_for_distribution_by_scalar(distribution, scalar=sigma, return_class=True)
        X = np.vstack([X, gaussian_sample])
    index_dict = {f"{round(X[i,0],4)}|{round(X[i,1],4)}":X[i,2] for i in range(X.shape[0])}
    np.random.shuffle(X)
    X_data = X[:,:2]
    y_data = X[:,2]
    return X_data, y_data, index_dict

class KMeansClustering:
    def __init__(self, n_clusters:int=3, max_iterations:int=100):
        self.X = None
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def get_euclidean_distance(self, center:np.ndarray, sample:np.ndarray):
        return np.sqrt(np.sum((center - sample)**2, axis=1))
    
    def fit(self, X:np.ndarray):
        self.X = X
        self.centroids = [np.random.choice(self.X, self.n_clusters, replace=False)]
        for _ in range(self.n_clusters-1):
            distances = np.sum([self.get_euclidean_distance(centroid, X) for centroid in self.centroids], axis=0)
            distances /= np.sum(distances)
            centroid_rev_index, = np.random.choice(range(len(X)), size=1, p=distances)
            self.centroids += [X[centroid_rev_index]]
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iterations:
            points_sorted = [[] for _ in range(self.n_clusters)]
            for point in X:
                distances = self.get_euclidean_distance(point, self.centroids)
                centroid_index = np.argmin(distances)
                points_sorted[centroid_index].append(point)
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in points_sorted]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def return_current_accuracy(self, true_labels:np.ndarray, class_mapping:dict=None):
        _, classification = self.return_centroids(self.X)
        if class_mapping is None:
            accuracy = (true_labels == classification).sum() / true_labels.shape[0]
        else:
            classification = [class_mapping[x] for x in classification]
            accuracy = (true_labels == classification).sum() / true_labels.shape[0]
        return accuracy

    def return_centroids(self, X):
        self.X = X
        centroids = []
        centroid_indicies = []
        for x in X:
            distances = self.get_euclidean_distance(x, self.centroids)
            centroid_index = np.argmin(distances)
            centroids.append(self.centroids[centroid_index])
            centroid_indicies.append(centroid_index)
        return centroids, centroid_indicies

def run_kmeans_and_gmm_experiment_for_k_clusters(k_clusters:int=3, plot:bool=False):
    results_dict = {}
    Px = {}
    distribution_labels = ('Pa', 'Pb', 'Pc')
    sigmas = [0.5, 1, 2, 4, 8]
    for sigma in sigmas:
        X = np.zeros(shape=(0,3)) # set to 0,3 if returning class, otherwise 0,2
        for distribution in distribution_labels:
            gaussian_sample = get_samples_for_distribution_by_scalar(distribution, scalar=sigma, return_class=True)
            Px[distribution] = gaussian_sample
            X = np.vstack([X, gaussian_sample])
        # np.random.shuffle(X)
        X_data = X[:,:2]
        y_data = X[:,2]
        kms = KMeans(n_clusters=k_clusters, n_init=10, init="k-means++", algorithm="lloyd").fit(X_data)
        gmm = GaussianMixture(n_components=k_clusters, n_init=30, covariance_type='diag').fit(X_data)
        # gmm = GaussianMixture(n_components=k_clusters, n_init=16).fit(X_data)
        gmm_accuracy = get_gmm_accuracy(gmm, sigma, X_data, y_data)
        kmc_accuracy = get_gmm_accuracy(kms, sigma, X_data, y_data)
        kmc_objective = round(objective(kms, kms.cluster_centers_, X_data),2)
        gmm_objective = round(objective(gmm, gmm.means_, X_data),2)
        print(f'gmm accuracy: {gmm_accuracy:.2%} (sigma: {sigma})')
        results_dict[sigma] = [sigma, kmc_accuracy, gmm_accuracy, kmc_objective, gmm_objective]
    if plot:
        plt.scatter(X_data[:,0], X_data[:,1], c=gmm.predict(X_data))
        plt.axis('equal')
        plt.show()
    for k, v in results_dict.items():
        print(f'{k}: {v}')
    return results_dict

def plot_kmc_gmm_results(data:list, plot_type:str=None, plot_title:str="K-Means Clustering vs Gaussian Mixture Model", savefile:str=None):
    x_values = [x[0] for x in data]
    match plot_type.lower()[:3]:
        case 'acc':
            label_y = 'Accuracy'
            kmc_y = [x[1] for x in data] 
            gmm_y = [x[2] for x in data]
        case 'obj':
            label_y = 'Objective'
            kmc_y = [x[3] for x in data]
            gmm_y = [x[4] for x in data]
        case other:
            return
    ### plot styling ###
    color_dict = {'plum':'#c8a4c6', 'lb':'#afd6d2'}
    title_font_dict = {'fontname':'GE Inspira Sans','fontsize':14}
    axes_label_fonts = {'fontname':'GE Inspira Sans','fontsize':11}
    plt.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
    plt.title(plot_title+'\n',**title_font_dict)
    kmeans_label = "K-Means Clustering"
    gmm_label = "Gaussian Mixture Model"
    plt.plot(x_values, kmc_y, label=kmeans_label, color=color_dict['lb'],linewidth=1.2) # KMC
    plt.plot(x_values, gmm_y, label=gmm_label, color=color_dict['plum'],linewidth=1.2) # GMM
    legend = plt.legend(facecolor='#fafbfd')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor('#2b2d2e')
    for text in legend.get_texts():
        text.set_color('#fafbfd')
        text.set_fontfamily('GE Inspira Sans')
    plt.ylabel(label_y,**axes_label_fonts)
    plt.xlabel('Variable Factor',**axes_label_fonts)
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e')
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

def import_dataset(dataset:str=None, to_numpy:bool=False, enumerate_headers:bool=False) -> pd.DataFrame:
    """use `dataset='2d'` to import 2d dataset and other for 1000d dataset,
    returns data as `DataFrame` with enumerated columns x_1 ... x_n if `enumerate_headers=True`,
    or optionally returns data as numpy ndarray if `to_numpy=True`"""
    read_params = {'dtype':'float64'}
    if '2' in dataset:
        filepath = './data/data2D.csv'
        if enumerate_headers:
            read_params['names'] = [f"x_{i+1}" for i in range(2)]
        else:
            read_params['header'] = None
            read_params['index_col'] = None
    else:
        filepath = './data/data1000D.csv'
        if enumerate_headers:
            read_params['names'] = [f"x_{i+1}" for i in range(1000)]
        else:
            read_params['header'] = None
            read_params['index_col'] = None
    df = pd.read_csv(filepath, **read_params)
    if to_numpy:
        df = df.to_numpy(dtype='float64', copy=False)
    return df

def plot_pca_analysis(original_data:pd.DataFrame, recon_data:pd.DataFrame, plot_title:str=None, savefile:str=None):
    ### plot styling ###
    color_dict = {'plum':'#b987b6', 'lb':'#afd6d2'} # normal plum: #c8a4c6, normal lb: #afd6d2
    title_font_dict = {'fontname':'GE Inspira Sans','fontsize':14}
    axes_label_fonts = {'fontname':'GE Inspira Sans','fontsize':12}
    if plot_title is None:
        plot_title = 'Analysis Plot'
    plt.title(plot_title+'\n',**title_font_dict)
    original_data_label = f"Original Data"
    recon_data_label = f"Reconstruction"
    if type(original_data) == np.ndarray:
        x_original = original_data[:,0]
        y_original = original_data[:,1]
    else:
        x_original = original_data[0]
        y_original = original_data[1]
    if type(recon_data) == np.ndarray:
        x_recon = recon_data[:,0]
        y_recon = recon_data[:,1]
    else:
        x_recon = recon_data[0]
        y_recon = recon_data[1]
    plt.scatter(x_original, y_original, label=original_data_label, c=color_dict['lb'], s=22, edgecolors='#2b2d2e', linewidths=0.2)
    plt.scatter(x_recon, y_recon, label=recon_data_label, c=color_dict['plum'], s=22, edgecolors='#2b2d2e', linewidths=0.2)
    legend = plt.legend(facecolor='#fafbfd')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor('#2b2d2e')
    for text in legend.get_texts():
        text.set_color('#fafbfd')
        text.set_fontfamily('GE Inspira Sans')
    plt.xlabel('X2 dimension',**axes_label_fonts) # set x label and styling
    plt.ylabel('X1 dimension',**axes_label_fonts) # set y label and styling
    plt.xticks(**axes_label_fonts) # set xticks styling
    plt.yticks(**axes_label_fonts) # set yticks styling
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e') # background color
    if savefile is not None:
        plt.savefig(savefile, dpi=399)
    plt.show()

def get_reconstruction_loss(input_matrix:np.ndarray, recon_matrix:np.ndarray) -> float:
    loss = np.sum((input_matrix - recon_matrix) ** 2, axis=1).mean()
    return loss

def master_PCA_by_type(input_matrix:pd.DataFrame, n_dim:int, pca_type:str=None, plot:bool=False, plot_title:str=None, savefile:str=None) -> dict:
    """use `pca_type='buggy'` for buggy PCA, `pca_type='demean'` for demeaned PCA, `pca_type='norm'` for normalized PCA,
    use `n_dim` to specify target number of dimensions to perform PCA analysis to, using `plot=True` to optionally plot results"""
    match pca_type.lower()[:2]:
        case 'bu':
            pca_type = 'buggy'
            mean = np.zeros(input_matrix.shape[1])
            std = np.ones(input_matrix.shape[1])
        case 'de':
            pca_type = 'demean'
            mean = input_matrix.mean(axis=0)
            std = np.ones(input_matrix.shape[1])
        case 'no':
            pca_type = 'norm'
            mean = input_matrix.mean(axis=0)
            std = input_matrix.std(axis=0)
        case 'dr':
            pca_type = 'dro'
        case other:
            print(f'Error: Unable to recognize argument provided to pca_type: {pca_type}')
            return
    if pca_type != 'dro':
        U, S, VT = np.linalg.svd((input_matrix-mean)/std)
        pca_components = VT[:n_dim]
        primary_pca = ((input_matrix-mean)/std).dot(pca_components.T)
        recon_matrix = (primary_pca.dot(pca_components)*std) + mean
        recon_error = MSE(input_matrix, recon_matrix)*(n_dim+1)
        recon_loss = np.sum((input_matrix - recon_matrix) ** 2, axis=1).mean()
        error_loss_quotient = round(recon_loss/recon_error,4)
        loss_check = get_reconstruction_loss(input_matrix,recon_matrix)
    else:
        std_scaler = SS(with_mean=True, with_std=False)
        recon_matrix = std_scaler.fit_transform(input_matrix)
        U, S, VT = np.linalg.svd(recon_matrix, full_matrices=False)
        S = np.diag(S)
        pca_components = recon_matrix.dot(VT.T[:,:n_dim])
        recon_matrix = std_scaler.inverse_transform(pca_components.dot(VT.T[:,:n_dim].T))
        recon_error = MSE(input_matrix, recon_matrix)*(n_dim+1)      
        recon_loss = np.sum((input_matrix - recon_matrix) ** 2, axis=1).mean()  
        error_loss_quotient = round(recon_loss/recon_error,4)
        loss_check = get_reconstruction_loss(input_matrix,recon_matrix)
    independent_loss = np.sum((input_matrix - recon_matrix) ** 2, axis=1).mean()
    print(f'performed {pca_type} PCA using n_dim={n_dim} with recon_error: {recon_error:.6f}, recon_loss: {recon_loss:.6f} ({error_loss_quotient}) -- loss_check: {loss_check:.6f}, ideal loss: {independent_loss:.6f}')
    if plot:
        plot_title = plot_title if plot_title is not None else 'Analysis Plot'
        plot_pca_analysis(input_matrix, recon_matrix, plot_title=f'{plot_title}, Reconstruction Error: {recon_error:.6f}', savefile=savefile)
    return {'input_matrix': input_matrix, 'recon_matrix': recon_matrix, 'recon_error': recon_error, 'recon_loss': recon_loss, 'i_matrix': recon_matrix}

def plot_generic_data(data, first_order_diff:bool=False, savefile:str=None, plot_title:str=None):
    x_values = [x[0] for x in data]
    y_values = [x[1] for x in data]
    if first_order_diff:
        y_values = np.abs(np.diff(y_values))
        x_values = [(i+1) for i in range(len(y_values))]
    color_dict = {'plum':'#c8a4c6', 'lb':'#afd6d2'} # normal plum: #c8a4c6, normal lb: #afd6d2, darker plum: #b987b6
    title_font_dict = {'fontname':'GE Inspira Sans','fontsize':14}
    axes_label_fonts = {'fontname':'GE Inspira Sans','fontsize':12}
    font_text_dict = {'fontname':'GE Inspira Sans','fontsize':10}
    if plot_title is None:
        plot_title = 'First Order Difference for 1K Dimension Data by Number Dimension'
    plt.title(plot_title+'\n',**title_font_dict)
    data_label = "First Order Difference (Abs)"
    vline_label = "D = 30, Pivot Dimension"      
    # plt.plot(np.arange(0,499,1), S[1:], label=data_label, color=color_dict['lb'])
    plt.plot(x_values, y_values, label=data_label, color=color_dict['lb'])
    plt.axvline(x=30, label=vline_label, color=color_dict['plum'], linestyle=':', linewidth=2.0)
    # plt.text(32,1200, "knee point at d=30", color=color_dict['plum'], fontdict=font_text_dict)
    legend = plt.legend(facecolor='#fafbfd')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor('#2b2d2e')
    for text in legend.get_texts():
        text.set_color('#fafbfd')
        text.set_fontfamily('GE Inspira Sans')
    plt.xlabel('Dimensions',**axes_label_fonts) # set x label and styling
    plt.ylabel('First Order Difference (Abs)',**axes_label_fonts) # set y label and styling
    plt.xticks(**axes_label_fonts) # set xticks styling
    plt.yticks(**axes_label_fonts) # set yticks styling
    plt.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e') # background color
    if savefile is not None:
        plt.savefig(savefile, dpi=399)
    plt.show()

def get_n_rand_matrix_rows(matrix:np.ndarray, n_indicies:int, rand_seed:int=None) -> np.ndarray:
    """selects `n_random` indicies at random to use for testing mean squared error loss between input and projected matrices"""
    if rand_seed is not None:
        np.random.seed(rand_seed)
    return matrix[np.random.choice(matrix.shape[0], n_indicies, replace=False), :]

dim1000_loss_data = [
    [1, 28022.637329]
    ,[2, 26404.251323]
    ,[3, 24939.283269]
    ,[4, 23590.914375]
    ,[5, 22246.043187]
    ,[6, 20933.439686]
    ,[7, 19720.96254]
    ,[8, 18583.028886]
    ,[9, 17468.072884]
    ,[10, 16362.306693]
    ,[11, 15269.297315]
    ,[12, 14216.896069]
    ,[13, 13229.120422]
    ,[14, 12278.426287]
    ,[15, 11333.527568]
    ,[16, 10416.761352]
    ,[17, 9503.866495]
    ,[18, 8631.341504]
    ,[19, 7782.704942]
    ,[20, 6958.161535]
    ,[21, 6158.370004]
    ,[22, 5404.710983]
    ,[23, 4670.558836]
    ,[24, 3965.905007]
    ,[25, 3296.933974]
    ,[26, 2644.009522]
    ,[27, 2010.326987]
    ,[28, 1388.635478]
    ,[29, 803.055238]
    ,[30, 273.045959]
    ,[31, 271.469457]
    ,[32, 269.874191]
    ,[33, 268.324708]
    ,[34, 266.750443]
    ,[35, 265.249603]
    ,[36, 263.759607]
    ,[37, 262.273757]
    ,[38, 260.778032]
    ,[39, 259.237794]
    ,[40, 257.811295]
    ,[41, 256.298009]
    ,[42, 254.924301]
    ,[43, 253.435229]
    ,[44, 251.979501]
    ,[45, 250.550607]
    ,[46, 249.162846]
    ,[47, 247.746508]
    ,[48, 246.41784]
    ,[49, 244.934667]
    ,[50, 244.522119]
    ,[51, 243.263275]
    ,[52, 241.785814]
    ,[53, 240.486208]
    ,[54, 239.067421]
    ,[55, 237.875352]
    ,[56, 236.532052]
    ,[57, 235.237808]
    ,[58, 233.941397]
    ,[59, 232.576443]
    ,[60, 231.276348]
    ,[61, 229.959032]
    ,[62, 228.717931]
    ,[63, 227.60441]
    ,[64, 226.193859]
    ,[65, 225.104305]
    ,[66, 223.601021]
    ,[67, 222.503335]
    ,[68, 221.296322]
    ,[69, 220.146202]
    ,[70, 218.788097]
    ,[71, 217.673873]
    ,[72, 216.449415]
    ,[73, 215.168565]
    ,[74, 213.783856]
    ,[75, 212.677674]
    ,[76, 211.415393]
    ,[77, 210.208955]
    ,[78, 209.243892]
    ,[79, 207.908922]
    ,[80, 206.741256]
    ,[81, 205.486098]
    ,[82, 204.326446]
    ,[83, 203.333071]
    ,[84, 202.047493]
    ,[85, 200.853034]
    ,[86, 199.65716]
    ,[87, 198.532035]
    ,[88, 197.516105]
    ,[89, 196.321425]
    ,[90, 195.247615]
    ,[91, 194.197842]
    ,[92, 193.084623]
    ,[93, 191.968894]
    ,[94, 190.854276]
    ,[95, 189.777996]
    ,[96, 188.702223]
    ,[97, 187.455768]
    ,[98, 186.190952]
    ,[99, 185.213357]
    ,[100, 184.151581]
]
