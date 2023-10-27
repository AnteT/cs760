import os, re, random
from collections import Counter
import numpy as np
import pandas as pd

def init_char_dict(alpha:float) -> dict:
    chars = 'abcdefghijklmnopqrstuvwxyz '
    return {c:alpha for c in chars}

def get_file_and_return_parsed(filename:str) -> str:
    with open(filename, 'r') as f:
        lines = f.read()
    lines = re.sub('[^a-zA-Z ]', '', lines)
    return lines

def get_thetas_for_langs(dirpath:str, display:bool=False) -> tuple[dict]:
    e_dict, j_dict, s_dict = init_char_dict(0.5), init_char_dict(0.5), init_char_dict(0.5)
    allfiles = os.listdir(dirpath)
    for filename in allfiles:
        if len(filename) == 6: # only 0-9e/j/s
            filepath = os.path.join(dirpath,filename)
            filechars = get_file_and_return_parsed(filepath)
            counts = Counter(filechars)
            match filename[0]:
                case 'e':
                    for k,v in counts.items():
                        e_dict[k] += v
                case 'j':
                    for k,v in counts.items():
                        j_dict[k] += v
                case 's':
                    for k,v in counts.items():
                        s_dict[k] += v
    # init theta dicts to return
    e_theta_dict,j_theta_dict,s_theta_dict = {}, {}, {}
    for language,l_dict in (('e',e_dict), ('j',j_dict), ('s',s_dict)):
        match language:
            case 'e':
                theta_dict = e_theta_dict
            case 'j':
                theta_dict = j_theta_dict
            case 's':
                theta_dict = s_theta_dict
        if display:
            print(f"{'-'*150}\n{language}\n{'-'*150}")
        lang_sum = sum(v for v in l_dict.values())
        for k,v in l_dict.items():
            c_theta = v/lang_sum
            theta_dict[k] = c_theta
            if display:
                print(f"{k}: {'{:.4f}'.format(c_theta)}")
    return e_theta_dict,j_theta_dict,s_theta_dict

def bag_of_words(filename:str, as_np:bool=True) -> list:
    data = get_file_and_return_parsed(filename)
    c_dict,c_counted = init_char_dict(alpha=0.0), Counter(data)
    for k,v in c_counted.items():
        c_dict[k] += v
    if as_np:
        e10_vector = np.array([x for x in c_dict.values()],dtype=np.int16) # as np array dtype int
    else:
        e10_vector = [int(x) for x in c_dict.values()] # as python list
    return e10_vector

def p_x_given_y(x_vector, thetas) -> float:
    total = 0
    for i in np.arange(len(x_vector)):
        total += x_vector[i] * np.log(thetas[i])
    return total

def init_thetas_with_alpha(X, y, alpha) -> np.ndarray:
    chars = 'abcdefghijklmnopqrstuvwxyz '
    langs = ('e','j','s')
    theta = np.empty([len(chars), len(langs)])
    for i, _ in enumerate(langs):
        l_bool = y == i
        c_count = np.sum(X[l_bool,:], axis = 0)
        cond_probs = (c_count + alpha) / (np.sum(c_count) + len(chars)*alpha)
        theta[:,i] = cond_probs
    return theta

def init_training_arrays(n_max:int=10):
    chars = 'abcdefghijklmnopqrstuvwxyz '
    l_dict = {'e': 0, 'j':1, 's':2}
    X = np.empty([3*n_max, 27], dtype=int)
    y = np.empty(3*n_max, dtype=int)
    for lang in l_dict.keys():
        for i in range(n_max):
            file_name = f"./data/languageID/{lang}{i}.txt"
            c_chars = open(file_name, "r").read()
            lang_num = l_dict[lang]
            idx = n_max * lang_num + i
            w_vector = [c_chars.count(char) for char in chars]
            X[idx,:] = w_vector
            y[idx] = lang_num
    return X, y

def generate_predictions_across_n(n_index:int=10):
    l_dict = {'e': 0, 'j':1, 's':2}
    X, y = init_training_arrays(n_max=10)
    theta = init_thetas_with_alpha(X, y, 0.5)
    p_pred = np.empty([3*n_index, 3])
    for l in ('e','j','s'):
        for i in range(n_index, n_index*2):
            file_name = f"./data/languageID/{l}{i}.txt"
            with open(file_name, 'r') as f:
                c_chars = f.read()
            l_idx = l_dict[l]
            idx = n_index * l_idx + i - n_index
            x_vector = np.array([c_chars.count(char) for char in 'abcdefghijklmnopqrstuvwxyz '])
            ll = np.sum(x_vector * np.log(theta).T, axis = 1)
            ll_normalized = ll - np.max(ll)
            p_probability = np.exp(ll_normalized)/np.sum(np.exp(ll_normalized))
            p_pred[idx,:] = p_probability
    print(np.round(p_pred, 2))

def generate_across_n_and_return_results(n_index:int=10, randomize:bool=False) -> pd.DataFrame:
    l_dict = {'e': 0, 'j':1, 's':2}
    X, y = init_training_arrays(n_max=10)
    theta = init_thetas_with_alpha(X, y, 0.5)
    df = pd.DataFrame(columns=['english','japanese','spanish','file'])
    for l in ('e','j','s'):
        for i in range(n_index, n_index*2):
            filename = f"./data/languageID/{l}{i}.txt"
            with open(filename, 'r') as f:
                c_chars = f.read()
            if randomize:
                c_chars = ''.join(random.sample(c_chars,len(c_chars)))
            l_idx = l_dict[l]
            idx = (n_index * l_idx) + (i - n_index)
            x_vector = np.array([c_chars.count(char) for char in 'abcdefghijklmnopqrstuvwxyz '])
            ll = np.sum(x_vector * np.log(theta).T, axis = 1)
            ll_normalized = ll - np.max(ll)
            p_probability = np.exp(ll_normalized)/np.sum(np.exp(ll_normalized))
            df.loc[idx,['english','japanese','spanish']] = [int(x) for x in np.round(p_probability,1)]
            df.loc[idx,'file'] = f"{l}{i}"
    print(df)

########################### run naive bayes functions ###########################
if __name__ == '__main__':
    e_theta_dict, j_theta_dict, s_theta_dict = get_thetas_for_langs("./data/languageID")
    x_vector = bag_of_words("./data/languageID/e10.txt")
    generate_across_n_and_return_results(n_index=10)
    generate_across_n_and_return_results(n_index=10, randomize=True)

"""
notes:

results from testing against sets ejs10-19:
┌────┬─────────┬──────────┬─────────┬─────────┐
│    │ english │ japanese │ spanish │ file    │
├────┼─────────┼──────────┼─────────┼─────────┤
│  1 │       1 │        0 │       0 │ e10.txt │
│  2 │       1 │        0 │       0 │ e11.txt │
│  3 │       1 │        0 │       0 │ e12.txt │
│  4 │       1 │        0 │       0 │ e13.txt │
│  5 │       1 │        0 │       0 │ e14.txt │
│  6 │       1 │        0 │       0 │ e15.txt │
│  7 │       1 │        0 │       0 │ e16.txt │
│  8 │       1 │        0 │       0 │ e17.txt │
│  9 │       1 │        0 │       0 │ e18.txt │
│ 10 │       1 │        0 │       0 │ e19.txt │
│ 11 │       0 │        1 │       0 │ j10.txt │
│ 12 │       0 │        1 │       0 │ j11.txt │
│ 13 │       0 │        1 │       0 │ j12.txt │
│ 14 │       0 │        1 │       0 │ j13.txt │
│ 15 │       0 │        1 │       0 │ j14.txt │
│ 16 │       0 │        1 │       0 │ j15.txt │
│ 17 │       0 │        1 │       0 │ j16.txt │
│ 18 │       0 │        1 │       0 │ j17.txt │
│ 19 │       0 │        1 │       0 │ j18.txt │
│ 20 │       0 │        1 │       0 │ j19.txt │
│ 21 │       0 │        0 │       1 │ s10.txt │
│ 22 │       0 │        0 │       1 │ s11.txt │
│ 23 │       0 │        0 │       1 │ s12.txt │
│ 24 │       0 │        0 │       1 │ s13.txt │
│ 25 │       0 │        0 │       1 │ s14.txt │
│ 26 │       0 │        0 │       1 │ s15.txt │
│ 27 │       0 │        0 │       1 │ s16.txt │
│ 28 │       0 │        0 │       1 │ s17.txt │
│ 29 │       0 │        0 │       1 │ s18.txt │
│ 30 │       0 │        0 │       1 │ s19.txt │
└────┴─────────┴──────────┴─────────┴─────────┘
[30 rows x 4 columns]
"""