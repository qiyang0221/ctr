import numpy as np
import pandas as pd
import pickle
import logging
import sys
from collections import Counter
from bistiming import Stopwatch

features = [
    'click', 'C1', 'C15', 'C16', 'C18', 'C20', 'banner_pos', 'site_category',
    'site_category', 'app_category', 'device_type', 'device_conn_type'
]

with Stopwatch('Read training data'):
    train_input_file = sys.argv[1]
    train = pd.read_csv(train_input_file)

    for feature_name in features:
        feature_value = set(train[feature_name].unique())
        with open('sets/'+feature_name+'.pkl', 'wb') as f:
            pickle.dump(feature_value, f)

    hours = set(range(24))
    with open('sets/hour.pkl','wb') as f:
        pickle.dump(hours,f)