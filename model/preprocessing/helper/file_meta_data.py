PCL_DATA_FILE_META = {
    'filepath': './data/raw/dontpatronizeme_pcl.tsv',
    'columns':['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label']
}

CATEGORIES_DATA_FILE_META = {
    'filepath': './data/raw/dontpatronizeme_categories.tsv',
    'columns':['par_id', 'art_id', 'text', 'keyword', 'country_code', 'span_start', 'span_finish', 'span_text', 'pcl_category', 'number_of_annotators']
}

TRAIN_SPLIT_FILE_META = {
    'filepath': './data/splits/train_semeval_parids-labels.csv',
}

OFFICIAL_DEV_SPLIT_FILE_META = {
    'filepath': './data/splits/dev_semeval_parids-labels.csv'
}