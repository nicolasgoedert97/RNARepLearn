import os
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#torch.set_num_threads(os.cpu_count())

print("Avalaible threads: "+str(torch.get_num_threads()))