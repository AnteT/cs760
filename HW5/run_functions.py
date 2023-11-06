from dep_functions import get_x_train_y_train_stacked_samples, KMeansClustering
from dep_functions import run_kmeans_and_gmm_experiment_for_k_clusters
from dep_functions import plot_kmc_gmm_results, plot_generic_data
from dep_functions import import_dataset
from dep_functions import master_PCA_by_type
from dep_functions import dim1000_loss_data

######################################## run main experiment ########################################
if __name__ == '__main__':
    centers = 3
    for sigma in (0.5, 1, 2, 4, 8):
        X_train, true_labels, label_dict = get_x_train_y_train_stacked_samples(sigma=sigma)
        kmc = KMeansClustering(n_clusters=centers)
        kmc.fit(X_train)
        class_centers, classification = kmc.return_centroids(X_train)
        kmc.return_current_accuracy()

    experiment_data = run_kmeans_and_gmm_experiment_for_k_clusters(k_clusters=3) # returns results data of experiment for both kmc and gmm
    plot_kmc_gmm_results(experiment_data, plot_type='accuracy', savefile='./data/figs/kmc-gmm-accuracy.png') # plots 2 side-by-side plots of objective for kmc vs gmm
    plot_kmc_gmm_results(experiment_data, plot_type='objective', savefile='./data/figs/kmc-gmm-objective.png') # plots 2 side-by-side plots of accuracy for kmc vs gmm

    plot_generic_data(dim1000_loss_data, savefile='./data/figs/mse-loss-d30.png') # supporting figures for d=30
    plot_generic_data(dim1000_loss_data, first_order_diff=True, savefile='./data/figs/mse-loss-fod-d30.png') # supporting figures for d=30

    input_matrix_2d = import_dataset(dataset='2d', to_numpy=True)
    master_PCA_by_type(input_matrix_2d, n_dim=1, pca_type='buggy', plot=False, plot_title='Buggy PCA', savefile='./data/figs/buggy-pca.png') # run and plot buggy PCA
    master_PCA_by_type(input_matrix_2d, n_dim=1, pca_type='demean', plot=False, plot_title='Demeaned PCA', savefile='./data/figs/demeaned-pca.png') # run and plot demeaned PCA
    master_PCA_by_type(input_matrix_2d, n_dim=1, pca_type='norm', plot=False, plot_title='Normalized PCA', savefile='./data/figs/normalized-pca.png') # run and plot normalized PCA
    master_PCA_by_type(input_matrix_2d, n_dim=1, pca_type='dro', plot=False, plot_title='DRO Analysis', savefile='./data/figs/dro-pca.png') # run and plot DRO analysis

    input_matrix_1000d = import_dataset(dataset='1000d', to_numpy=True)
    master_PCA_by_type(input_matrix_1000d, n_dim=30, pca_type='buggy', plot=False, savefile=None) # buggy pca on chosen d=30
    master_PCA_by_type(input_matrix_1000d, n_dim=30, pca_type='demean', plot=False, savefile=None) # demeaned pca on chosen d=30
    master_PCA_by_type(input_matrix_1000d, n_dim=30, pca_type='norm', plot=False, savefile=None) # normalized pca on chosen d=30
    master_PCA_by_type(input_matrix_1000d, n_dim=30, pca_type='dro', plot=False, savefile=None) # dro on chosen d=30

