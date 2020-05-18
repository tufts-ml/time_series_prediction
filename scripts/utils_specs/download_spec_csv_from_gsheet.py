import argparse
import requests
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    with open(args.config_json_path, 'r') as f:
        gsheet_info = json.load(f)

    print("Downloading sheets from provided URL")
    for gid, sheet_name in zip(gsheet_info['spec_gid_list'], gsheet_info['spec_sheet_name_list']):
        sheet_url = gsheet_info['spec_gsheet_url_pattern']
        for var, val in gsheet_info.items():
            if sheet_url.count("{{%s}}" % var):
                sheet_url = sheet_url.replace("{{%s}}" % var, val)
        for var, val in [('gid', gid), ('sheet_name', sheet_name)]:
            if sheet_url.count("{{%s}}" % var):
                sheet_url = sheet_url.replace("{{%s}}" % var, val)
        ans = requests.get(sheet_url)
        ans.raise_for_status()

        csv_str = ans.content.decode('utf-8')
        out_csv_path = os.path.join(
            args.output_dir,
            gsheet_info['output_csv_path_pattern'].replace("{{sheet_name}}", sheet_name)
            )
        with open(out_csv_path, 'w') as f:
            for line in csv_str.split('\n'):
                f.write("%s\n" % line)
        print("... wrote sheet %s to %s" % (sheet_name, out_csv_path))


