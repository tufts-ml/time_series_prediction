import json
def load_data_dict_json(data_dict_file):
    with open(data_dict_file, 'r') as f:
        data_dict = json.load(f)
        try:
            data_dict['fields'] = data_dict['schema']['fields']
        except KeyError:
            pass
    return data_dict

def merge_data_dicts(data_dicts_list):
    # get a single consolidated data dict for all features and another for outcomes
    # combine all the labs, demographics and vitals jsons into a single json
    features_data_dict = dict()
    features_data_dict['schema']= dict()
    
    features_dict_merged = []
    for data_dict in data_dicts_list:
        features_dict_merged += data_dict['schema']['fields']  
    
    feat_names = list()
    features_data_dict['schema']['fields'] = []
    for feat_dict in features_dict_merged:
        if feat_dict['name'] not in feat_names:
            features_data_dict['schema']['fields'].append(feat_dict)
            feat_names.append(feat_dict['name'])
    return features_data_dict