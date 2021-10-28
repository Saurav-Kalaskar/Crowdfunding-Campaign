"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

with open("model_pickle.sav", "rb") as f:
    logistic = pickle.load(f)

feature_names = ['goal', 'staff_pick_True',
       'cam_length', 'blurb_length', 'country_US', 'main_category_comics',
       'main_category_crafts', 'main_category_dance', 'main_category_design',
       'main_category_fashion', 'main_category_film & video',
       'main_category_food', 'main_category_games', 'main_category_journalism',
       'main_category_music', 'main_category_photography',
       'main_category_publishing', 'main_category_technology',
       'main_category_theater']



def make_api_prediction(feature_dict):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns a dictionary with the following keys
      all_probs: a list of dictionaries with keys 'name', 'prob'. This tells the
                 probability of class 'name' appearing is the value in 'prob'
      most_likely_class_name: string (name of the most likely class)
      most_likely_class_prob: float (name of the most likely probability)
    """
    x_input = [feature_dict[name] for name in feature_names]
    x_input = [0 if val == '' else float(val) for val in x_input]

    pred_probs = logistic.predict([x_input])

   # probs = [{'name': logistic.target_names[index], 'prob': pred_probs[index]}
             #for index in np.argsort(pred_probs)[::-1]]

    response = int(pred_probs[0])
    
    #{
        #'all_probs': probs,
        #'most_likely_class_name': probs[0]['name'],
        #'most_likely_class_prob': probs[0]['prob'],
    #}

    #return response
        
    #response = pred_outcome
    
    #{
        #'all_probs': probs,
       # 'most_likely_class_name': probs[0]['name'],
        #'most_likely_class_prob': probs[0]['prob'],
    #}

    return response

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what setting all params to 0 predicts")
    features = {f:'0' for f in feature_names}
    print('Features are')
    pprint(features)

    response = make_api_prediction(features)
    print("The returned object")
    pprint(response)
