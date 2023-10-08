from dep_functions import plot_roc, D2z_1NN, nearest_neighbor_by_k, begin_training, five_fold_CV_across_varying_K, knn_vs_logistic_regression, run_training_cycle

############################## run main function ##############################
plot_roc(savefile='HW3/data/solutions/spam-roc-curve.png',color='#afd6d2') # create roc plot (3.a)
D2z_1NN(filename='HW3/data/D2z.txt',plot=True,savefile='HW3/data/solutions/d2z-1nn-plot.png') # create 1NN plot (2.1)
nearest_neighbor_by_k(top_k=1, fold=5) # create 5 fold cross validation emails.csv (2.2)
begin_training(num_iter=1_000, lr=0.01, k_fold=5, randomize=False) # create gradient descent from scratch (2.3)
run_training_cycle(num_iter=5_000,lr=0.005,test_train_split=0.2, randomize=True) # recreated logistic regression without cv to try and improve recall, not related to any question in particular
five_fold_CV_across_varying_K(savefile='knn-across-k.png') # plot five-fold cross validation across varying K (4)
knn_vs_logistic_regression(savefile='HW3/data/solutions/knn-vs-lr.png') # knn vs logistic regression plot (5)

"""
notes:

results for nearest_neighbor_by_k:
-----------------------------------
fold  accuracy  precision    recall
   1     0.825   0.654494  0.817544
   2     0.853   0.685714  0.866426
   3     0.862   0.721212  0.838028
   4     0.851   0.716418  0.816327
   5     0.775   0.605744  0.758170
-----------------------------------

# best results after recreating logistic regression approach with more control to attempt to reduce recall:

run_training_cycle(iterations=5_000,lr=0.005,test_train_split=0.2,randomize=True)
-----------------------------------------------------------------
       type  total  correct  incorrect accuracy precision  recall
0  training   4000     3828        172   95.70%    94.49%  90.18%
1      test   1000      911         89   91.10%    87.54%  82.68%
-----------------------------------------------------------------
"""