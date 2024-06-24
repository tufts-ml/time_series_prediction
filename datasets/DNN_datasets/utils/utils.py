import json
def load_data_dict_json(data_dict_file):
    with open(data_dict_file, 'r') as f:
        data_dict = json.load(f)
        try:
            data_dict['fields'] = data_dict['schema']['fields']
        except KeyError:
            pass
    return data_dict


def parse_id_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'id' or 
                              col['role'] == 'key'):
            cols.append(col['name'])
    return cols

def parse_output_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'outcome' or 
                              col['role'] == 'output'):
            cols.append(col['name'])
    return cols

def parse_feature_cols(data_dict):
    non_time_cols = []
    for col in data_dict['fields']:
        if 'role' in col and col['role'] in ('feature', 'measurement', 'covariate'):
            non_time_cols.append(col['name'])
    non_time_cols.sort()
    return non_time_cols

def parse_time_cols(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        if (col['name'] == 'hours' or col['role'].count('time') > 0):
            time_cols.append(col['name'])
    return time_cols
