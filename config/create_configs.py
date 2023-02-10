import argparse
import ast
import itertools
import os


def parse_entry(entry):
    value = entry[1].strip()

    if value.startswith("[") and value.endswith("]"):
        value = ast.literal_eval(value)
    
    elif value.isdigit():
        value = int(value)
    
    return entry[0].strip(), value

def dict_to_gin(dic, output):
    with open(output,"w") as out:
        for name, value in dic.items():
            out.write(str(name)+" = "+str(value)+"\n")

def read_gin(gin, toList):
    config = {}
    for line in open(gin, "r"):
        if len(line)>1 and not line.startswith("#"):

            entry = line.split("=")
            name, value = parse_entry(entry)
            if toList and not isinstance(value, list):
                value = [value]
            config[name] = value
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--base-gin')
parser.add_argument('--grid-gin')
parser.add_argument('--output-path')

args = parser.parse_args()

output_path = args.output_path

if not os.path.exists(output_path):
    os.makedirs(os.path.join(output_path, "gin"))


base_config = read_gin(args.base_gin, False)
grid_config = read_gin(args.grid_gin, True)


for param_vals in itertools.product(*grid_config.values()):
    params = dict(zip(grid_config.keys(), param_vals))
    output_name= "config_"+"_".join([str(i.split(".")[-1])+str(j) for i,j in params.items()])+".gin"
    output = params | base_config
    dict_to_gin(output, os.path.join(output_path,"gin",output_name))

with open(os.path.join(output_path,".done"), "w") as end:
    end.close()



