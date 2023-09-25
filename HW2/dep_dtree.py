### dependencies for dtree functions ###
import pandas as pd
import numpy as np
from math import log2
from rich import print as cprint
from anytree import Node, RenderTree
import pickle, os
from typing import Dict

def iterate_through_tree(df:pd.DataFrame, parent_id=None, features=['x_1','x_2'], label='y', history=None, current_step=0, prior_node='', show_branches=False):
    df_results = pd.DataFrame(columns=['branch','column','candidate','ig_ratio','ig'])
    if history is None:
        history = []
    best_ig = 0
    best_col = ''
    best_candidate = 0
    for feature in features:
        for candidate_split in df[feature].unique().tolist():
            values = get_infogain_candidate_split(df, feature,candidate_split, label=label) # format: (branch, split_column, split_candidate, info_gain_ratio, info_gain)
            df_results.loc[len(df_results.index)] = list(values)
            split_branch, split_column, split_candidate, info_gain_ratio, info_gain = values
            if info_gain_ratio != np.NaN:
                if info_gain_ratio > best_ig:
                    best_ig = info_gain_ratio
                    best_col = split_column
                    best_candidate = split_candidate

    # print(df_results)
    if show_branches:
        print(f'best col: {best_col}, candidate: {best_candidate}, info gain ratio: {best_ig}')
    df_left_split = df[df[best_col] >= best_candidate]
    df_right_split = df[df[best_col] < best_candidate]
    branches_dict = {'left':df_left_split,'right':df_right_split}
    for direction, df_branch in branches_dict.items():
        branch_length = len(df_branch.index)
        branch_ent = get_entropy(df_branch[label])
        if branch_ent <= 0 or branch_length <= 0:
            if show_branches:
                print(f'\n{"-"*54} {direction.upper()} LEAF CREATED {"-"*54}')
                print(f"branch dataframe:\n{'-'*40}\n{df_branch}\n{'-'*40}")
            child_id = f'{best_col} >= {best_candidate}' if direction == 'left' else f'{best_col} < {best_candidate}'
            # path_trace = f"""{current_step} ({direction}): LEAF created with {f'{best_col} >= {best_candidate}' if direction == 'left' else f'{best_col} < {best_candidate}' } with ig_ratio: {'{:.3f}'.format(best_ig)}"""
            df_y = df_branch[label].value_counts().rename_axis(f'{label}_label').reset_index(name='counts')
            df_y = df_y.sort_values(by=[f'{label}_label'],ascending=False)
            df_y = df_y.sort_values(by=['counts'],ascending=False)
            predict_y = df_y.iloc[0,0]
            child_node = Node(child_id, parent=parent_id, type='leaf', col=best_col, value=best_candidate, direction=direction, prediction=predict_y)
            prior_node += f' {current_step} left -> leaf: predict y={predict_y};' if direction == 'left' else f' {current_step} right -> leaf: predict {label}={predict_y};'
            path_trace = f"""{current_step} ({prior_node}): LEAF created with {f'{best_col} >= {best_candidate}' if direction == 'left' else f'{best_col} < {best_candidate}' } with ig_ratio: {'{:.3f}'.format(best_ig)}"""
            history.append(path_trace)
            if show_branches:
                print(f"{path_trace}:\n{df_y}")
            continue
        else:
            if show_branches:
                print(f'\n{"-"*54} {direction.upper()} NODE CREATED {"-"*54}')
                print(f"branch dataframe:\n{'-'*40}\n{df_branch}\n{'-'*40}")
            child_id = f'{best_col} >= {best_candidate}' if direction == 'left' else f'{best_col} < {best_candidate}'
            child_node = Node(child_id, parent=parent_id, type='node', col=best_col, value=best_candidate, direction=direction, prediction=None)
            prior_node += f' {current_step} left -> node;' if direction == 'left' else f' {current_step} right -> node;'
            current_step += 1
            path_trace = f"""{current_step} ({prior_node}): NODE created with {f'{best_col} >= {best_candidate}' if direction == 'left' else f'{best_col} < {best_candidate}' } with ig_ratio: {'{:.3f}'.format(best_ig)}"""
            history.append(path_trace)
            if show_branches:
                print(f"{path_trace}\n")
            iterate_through_tree(df_branch, parent_id=child_node,features=['x_1','x_2'], history=history, current_step=current_step, prior_node=prior_node)
    # return parent_id

def import_druns_as_df():
    with open('HW2/data/Druns.txt') as f:
        lines = f.readlines()
    num_rows=len(lines)  
    headers = ['x_1', 'x_2', 'y']
    raw_data = []
    for row in range(num_rows):
        tmp_row = lines[row].strip('\n').split(' ')
        tmp_row = [float(x) for x in tmp_row]
        raw_data.append(tmp_row)
    df = pd.DataFrame(data=raw_data, columns=headers)
    df['y'] = df['y'].astype('Int64')
    return df

def get_entropy(labels) -> float:
    """calculates and returns the labels entropy, `labels` arg can be of any valid iterable"""
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    for i in probs:
        ent -= i * log2(i)
    return ent

def get_infogain_candidate_split(df:pd.DataFrame, split_column:str = 'x_1', split_candidate:float = 6.0, label:str='y', verbose:bool=False):
    ent_dataset = get_entropy(df[label])
    length_dataset = len(df.index)
    df_lh = df[df[split_column] >= split_candidate]
    length_lh = len(df_lh.index)
    df_rh = df[df[split_column] < split_candidate]
    length_rh = len(df_rh.index)
    if (length_lh < 1 or length_rh < 1):
        info_gain = 0
        info_gain_ratio = np.NaN
        if not verbose:
            return ('leaf',split_column, split_candidate, info_gain_ratio, info_gain)
        else:
            if length_lh >= 1:
                ent_lh = "{:.4f}".format(ent_dataset)
                ent_rh = 'nan'
            elif length_rh >= 1:
                ent_rh = "{:.4f}".format(ent_dataset)
                ent_lh = 'nan'
            entropy_dict = {'label':f"{split_column} >= {split_candidate}",'info_gain_ratio':"{:.4f}".format(info_gain_ratio), 'info_gain':"{:.4f}".format(info_gain),'ent_dataset': "{:.4f}".format(ent_dataset), 'ent_lh': ent_lh, 'ent_rh': ent_rh}
            return ('leaf',split_column, split_candidate, info_gain_ratio, info_gain, entropy_dict)
    ent_lh = get_entropy(df_lh[label])
    ent_rh = get_entropy(df_rh[label])
    info_gain = ent_dataset-(((length_lh/length_dataset)*(ent_lh)) + ((length_rh/length_dataset)*(ent_rh)))
    info_gain_ratio = info_gain/(-(((length_lh/length_dataset)*log2(length_lh/length_dataset))+((length_rh/length_dataset)*log2(length_rh/length_dataset))))
    if not verbose:
        return ('node',split_column, split_candidate, info_gain_ratio, info_gain)
    else:
        entropy_dict = {'label':f"{split_column} >= {split_candidate}",'info_gain_ratio':"{:.4f}".format(info_gain_ratio), 'info_gain':"{:.4f}".format(info_gain),'ent_dataset': "{:.4f}".format(ent_dataset), 'ent_lh': "{:.4f}".format(ent_lh), 'ent_rh': "{:.4f}".format(ent_rh)}
        return ('node',split_column, split_candidate, info_gain_ratio, info_gain, entropy_dict)

def import_txt_as_dataframe(filename) -> pd.DataFrame:
    df_D1 = pd.read_csv(filename,sep=' ', names=['x_1','x_2','y'])
    return df_D1

def crawl_generate_tree(df:pd.DataFrame, features:list=['x_1','x_2'], label:str='y', verbose:bool=False):
    """optional kwargs like `features` and `label` are setup as sensible defaults for the cs760 datasets
    \nuse `verbose = True` if you want all outputs at every step of the process (ie why node vs leaf)
    \notherwise, use `crawl_generate_tree(Dataframe)` to end up with terminal ascii tree:
    \n  root
    \n  ├── x_1 >= 10 (predict y = 1)
    \n  └── x_1 < 10
    \n      ├── x_2 >= 3 (predict y = 1)
    \n      └── x_2 < 3 (predict y = 0)
    """
    history = []
    root_node = Node("root", type='root', col=None, value=None, direction=None, prediction=None)
    iterate_through_tree(df, parent_id=root_node, features=features, label=label, history=history, show_branches=verbose)
    terminal_width = os.get_terminal_size()[0]
    cprint(f'[bold #efac65]{"-"* terminal_width}[/]')
    ### nodes have attributes of name (candidate split chosen) & type (node | leaf) & prediction (label | None) ###
    num_nodes = 0
    num_leaves = 0
    for pre, _, node in RenderTree(root_node):
        split = node.name
        branch_type = node.type
        col = node.col # for model traversal if needed
        value = node.value # for model traversal if needed
        direction = node.direction # for model traversal if needed
        result = f""" ({label} = {node.prediction})""" if node.prediction is not None else ''
        if branch_type == 'leaf':
            num_leaves += 1
            cprint(f"""[bold white]{pre}[/][bold #b2f2bb]{split}[/][bold #b2f2bb]{result}[/]""")
            # print(col, value, direction)
        else:
            num_nodes += 1
            cprint(f"""[bold white]{pre}[/][bold #a5d8ff]{split}[/][bold #b2f2bb]{result}[/]""")
            # print(col, value, direction)
    cprint(f'\n[bold white]branches: [/][bold #a5d8ff]{num_nodes} nodes[/] [bold white]&[/] [bold #b2f2bb]{num_leaves} leaves[/]')
    cprint(f'[bold #efac65]{"-"* terminal_width}[/]')
    return root_node

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

def verbose_tree_model_walk(root:Node, x_1, x_2) -> int:
    if root.prediction is not None:
        return root.prediction
    left_node = None
    right_node = None
    target_col = None
    target_value = None
    children_nodes = root.children
    for child in children_nodes:
        target_col = child.col
        target_value = child.value
        if child.direction == 'left':
            left_node = child
        elif child.direction == 'right':
            right_node = child
    if target_col == 'x_1':
        if x_1 >= target_value:
            return verbose_tree_model_walk(left_node, x_1, x_2)
        elif x_1 < target_value:
            return verbose_tree_model_walk(right_node, x_1, x_2)
    elif target_col == 'x_2':
        if x_2 >= target_value:
            return verbose_tree_model_walk(left_node, x_1, x_2)
        elif x_2 < target_value:
            return verbose_tree_model_walk(right_node, x_1, x_2)

def validate_sample_against_model(root:Node, df_validate:pd.DataFrame):
    total = len(df_validate.index)
    correct = 0
    wrong = 0
    for row in df_validate.itertuples():
        x_1, x_2, y = row.x_1, row.x_2, row.y
        prediction = traverse_tree_model(root,x_1,x_2)
        if prediction == y:
            correct += 1
        else:
            wrong += 1
    print(f'validation completed:\ntotal: {total}\ncorrect: {correct}\nwrong: {wrong}\nerror rate: {"{:.4f}".format(1-(correct/total))}')

def generate_train_and_validation_set_for_Dbig(filename='HW2/data/Dbig.txt') -> dict[pd.DataFrame,pd.DataFrame]:
    """returns the Dbig.txt data as a train and test dict"""
    df = pd.read_csv(filename,sep=' ', names=['x_1','x_2','y'])
    df = df.sample(frac=1) # shuffle
    df_train = df.iloc[:8192]
    df_train.to_csv('HW2/data/Dbig-train.csv')
    df_test = df.iloc[8192:]
    df_test.to_csv('HW2/data/Dbig-test.csv')
    print(len(df_train.index))
    print(len(df_test.index))
    return {'train':df_train, 'test':df_test}

def get_Dbig_from_csv(csv='Dbig-train.csv'):
    df = pd.read_csv(csv,index_col=0)
    return df

def summarize_training_results():
    headers = ('set', 'nodes', 'error')
    data = ((32, 9, 0.2024),(128, 12, 0.0658),(512, 19, 0.0592),(2048, 60, 0.0409),(8192, 137, 0.0155))
    df = pd.DataFrame(data=data, columns=headers)
    print(df)

def load_model(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(filename)
    return obj

def save_model(obj, filename):
    with open(filename,"wb") as f:
        pickle.dump(obj,f)

def import_train_display_tree(filename:str, savename:str=None) -> Dict[pd.DataFrame,Node]:
    """imports `filename` arg, trains a decision tree, displays it, and optionally saves to .pkl file if `savename != None` usage:
    \n`tree_root = import_train_display_tree('HW2/data/D2.txt', savename='D2.pkl')`
    \nreturns the a dict containing the training data and the resulting model for the tree"""
    filelabel, filetype = filename.split('.')
    filelabel = filelabel.split('\\')[-1]
    print(f'beginning process for {filelabel.split("/")[-1]}')
    if filetype == 'csv':
        df = pd.read_csv(filename,index_col=0)
    else:
        df = pd.read_csv(filename,names=['x_1','x_2','y'],sep=' ')
    root = crawl_generate_tree(df)
    if savename is not None:
        if '.pkl' not in savename:
            savename = f'{filelabel}.pkl'
        save_model(root,savename)
    print(f'successfully saved root node model as: "{savename}"')
    return {'data':df,'model':root}
