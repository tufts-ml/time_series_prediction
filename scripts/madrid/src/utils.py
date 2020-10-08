import json
def load_data_dict_json(data_dict_file):
    with open(data_dict_file, 'r') as f:
        data_dict = json.load(f)
        try:
            data_dict['fields'] = data_dict['schema']['fields']
        except KeyError:
            pass
    return data_dict
