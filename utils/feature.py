
def num_features():
    return ['abnormal_period', 'hour']

def cat_features():
    return ['weekday', 'month']

def column_feature():
    return num_features() + cat_features()