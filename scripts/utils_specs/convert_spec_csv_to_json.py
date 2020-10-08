import argparse
import pandas as pd
import json
import copy
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--row_template_json", type=str, default='row_template.json')
    parser.add_argument("--sheet_template_json", type=str, default='sheet_template.json')
    args = parser.parse_args()

    with open(args.config_json_path, 'r') as f:
        config = json.load(f)

    with open(args.row_template_json, 'r') as f:
        row_template = json.load(f)

    with open(args.sheet_template_json, 'r') as f:
        sheet_template = json.load(f)

    for gid, sheet_name, csv_filename in zip(
            config['spec_gid_list'],
            config['spec_sheet_name_list'],
            config['spec_csv_filename_list']):
        
        sheet = copy.deepcopy(sheet_template)
        sheet['name'] = sheet['name'].replace("{{sheet_name}}", sheet_name)
        sheet['path'] = sheet['path'].replace("{{csv_filename}}", csv_filename)

        out_csv_path = os.path.join(
            args.output_dir,
            config['output_csv_path_pattern'].replace("{{sheet_name}}", sheet_name)
            )
        out_json_path = os.path.join(
            args.output_dir,
            config['output_json_path_pattern'].replace("{{sheet_name}}", sheet_name)
            )
        csv_df = pd.read_csv(out_csv_path, dtype=str)
        row_list = []
        for rowid, row_df in csv_df.iterrows():
            row = copy.deepcopy(row_template)
            for k, v in row_template.items():
                if isinstance(v, dict):
                    v = v.__repr__()
                    isdict = True
                else:
                    isdict = False
                assert isinstance(v, str)
                while v.count("{{") > 0:
                    start = v.find("{{")
                    stop = v.find("}}", start)
                    varname = v[start+2:stop]
                    v = v.replace("{{%s}}" % varname, str(row_df[varname]))
                if isdict:
                    row[k] = json.loads(v.replace("'", '"'))
                else:
                    row[k] = v
            row_list.append(row)

        sheet['schema']['fields'] = row_list

        sheet = json.dumps(sheet, indent=4, sort_keys=False)
        with open(out_json_path, 'w') as f:
            f.write(sheet)
        print("Wrote to file: %s" % out_json_path)
