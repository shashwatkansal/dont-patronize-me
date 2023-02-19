'''
The file where the raw DPM PCL data is stored
'''
PCL_DATA_FILE_META = {
    'filepath': './data/raw/dontpatronizeme_pcl.tsv',
    'columns':['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label']
}

'''
The file where the raw DPM category data is stored
'''
CATEGORIES_DATA_FILE_META = {
    'filepath': './data/raw/dontpatronizeme_categories.tsv',
    'columns':['par_id', 'art_id', 'text', 'keyword', 'country_code', 'span_start', 'span_finish', 'span_text', 'pcl_category', 'number_of_annotators']
}

'''
The file where the training data split by paragraph id (par_id) is stored
'''
TRAIN_SPLIT_FILE_META = {
    'filepath': './data/splits/train_semeval_parids-labels.csv',
}

'''
The file where the dev data split by paragraph id (par_id) is stored
'''
OFFICIAL_DEV_SPLIT_FILE_META = {
    'filepath': './data/splits/dev_semeval_parids-labels.csv'
}

'''
The file where the dev pcl data is stored
'''
OFFICIAL_DEV_PCL_DATA_FILE_META = {
    'filepath': './data/dontpatronizeme_pcl_dev.tsv'
}

'''
The file where the training pcl data is stored
'''
TRAINING_PCL_DATA_FILE_META = {
    'filepath': './data/dontpatronizeme_pcl_training.tsv'
}

