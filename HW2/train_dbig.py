from dep_dtree import crawl_generate_tree, validate_sample_against_model, generate_train_and_validation_set_for_Dbig

############################ run main function ############################
if __name__ == '__main__':
    ### retrain all models using permutations ###
    dbig_dict = generate_train_and_validation_set_for_Dbig('HW2/data/Dbig.txt')
    df_validation = dbig_dict['test']
    df_train = dbig_dict['train']
    df_32 = df_train.iloc[:32]
    df_128 = df_train.iloc[:128]
    df_512 = df_train.iloc[:512]
    df_2048 = df_train.iloc[:2048]
    df_8192 = df_train.iloc[:8192]
    df_dict = {"df_32":df_32,"df_128":df_128,"df_512":df_512,"df_2048":df_2048,"df_8192":df_8192}
    for k,v in df_dict.items():
        print(f'\nbeginning on dataset {k}...')
        root = crawl_generate_tree(v)
        # save_model(root,f'{k}.pkl')
        validate_sample_against_model(root,df_validation)
        print(f'finished dataset {k}')