### plot decision bounds for a given model ###
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from anytree import Node
import pickle

def load_model(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def traverse_tree_model(root:Node, x_1, x_2) -> int:
    if root.prediction is not None:
        return root.prediction
    left_node, right_node = root.children[0], root.children[1]
    target_col, target_value = left_node.col, left_node.value
    if target_col == 'x_1':
        if x_1 >= target_value:
            return traverse_tree_model(left_node, x_1, x_2)
        return traverse_tree_model(right_node, x_1, x_2)
    else:
        if x_2 >= target_value:
            return traverse_tree_model(left_node, x_1, x_2)
        return traverse_tree_model(right_node, x_1, x_2)
    
def evaluate_model_from_pkl_and_return_dataframe(model:str='df_32', df:pd.DataFrame=None) -> pd.DataFrame:
    if '.pkl' not in model:
        model = f"{model}.pkl"
    root = load_model(model)
    df['y'] = np.NaN
    for idx,row in enumerate(df.itertuples()):
        x_1, x_2 = row.x_1, row.x_2
        y = traverse_tree_model(root, x_1, x_2)
        df.loc[idx,'y'] = y
    df['y'] = df['y'].astype('Int64')
    return df

def draw_decision_boundary(model:str=None, min_x:float=-1.5, max_x:float=1.5, title=None, savefile=None):
    # colors=['#c8a4c6','#afd6d2'] # format of y=0 color (plum), y=1 color (lightblue)
    colors=['#91678f','#afd6d2'] # format of y=0 color (darker plum), y=1 color (lightblue)
    xval = np.linspace(min_x,max_x,50).tolist()
    xdata = []
    for i in range(len(xval)):
        for j in range(len(xval)):
            xdata.append([xval[i],xval[j]])
    df = pd.DataFrame(data=xdata,columns=['x_1','x_2'])
    df = evaluate_model_from_pkl_and_return_dataframe(model=model,df=df)
    
    ### integration point ###
    d_columns = df.columns.to_list()
    d_label = d_columns[-1]
    d_xfeature = d_columns[0]
    d_yfeature = d_columns[1]
    df = df.sort_values(by=d_label)
    d_xlabel = f"feature  $\mathit{{{d_xfeature}}}$"
    dy_ylabel = f"feature  $\mathit{{{d_yfeature}}}$"
    fontfamily = {'fontname':'GE Inspira Sans'}
    plt.xlabel(d_xlabel, fontsize=13, **fontfamily)
    plt.ylabel(dy_ylabel, fontsize=13, **fontfamily)
    legend_labels = []
    for i,label in enumerate(df[d_label].unique().tolist()):
        df_set = df[df[d_label]==label]
        print(f"count for dataset {d_label} = '{label}': {len(df_set.index)}")
        set_x = df_set[d_xfeature]
        set_y = df_set[d_yfeature]
        plt.scatter(set_x,set_y,c=colors[i],marker='s', s=50)
        legend_labels.append(f"""{d_label} = {label}""")
    if title is None:
        title = f"""$\mathit{{D_{{{model.split('_')[-1]}}}}}$ """ + "Decision Boundary Visualization" + '\n'
    if not title.endswith('\n'):
        title += '\n'
    plt.title(title, fontsize=13, **fontfamily)
    ax = plt.gca()
    ax.set_facecolor('#2b2d2e') # background of plot
    plt.legend(legend_labels)
    model = model.split('.')[0]
    savefile = f"""HW2/data/solutions/{model}-decision-boundary.png"""
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()     

def model_y(row):
    x_1, x_2 = row.x_1, row.x_2
    if row.x_1 >= .209:
        return 0
    return 1

############################## run main function ##############################
### D1.txt after pkl file generated in train_dtree.py ###
draw_decision_boundary(model='D1.pkl', min_x=0, max_x=1.0, title="D1 Decision Boundary Visualization")

### D2.txt after pkl file generated in train_dtree.py ###
draw_decision_boundary(model='D2.pkl', min_x=0, max_x=1.0, title="D2 Decision Boundary Visualization")

### D32, D128, D512, D2048, D8196 models after trained in train_dbig.py ###
models = ("df_32","df_128","df_512","df_2048","df_8192")
for model in models:
    draw_decision_boundary(model=model, min_x=-1.5, max_x=1.5, subject="Decision Boundary Visualization")