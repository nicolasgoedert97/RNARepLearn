import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-base', required=True)
parser.add_argument('-files_list', required=True)

args = parser.parse_args()

ids = []
with open(args.files_list, "r") as flist:
    for id in flist:
        ids.append(id.strip())
        break

print(ids)