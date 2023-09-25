from dep_dtree import import_train_display_tree, validate_sample_against_model

if __name__ == '__main__':
    datasets = ('HW2/data/Druns.txt', 'HW2/data/D1.txt', 'HW2/data/D2.txt') # Dbig.txt is trained in separate file: train_dbig.py
    for dataset in datasets:
        data_dict = import_train_display_tree(dataset)
        model, df = data_dict['model'], data_dict['data']
        validate_sample_against_model(model,df)