import torch
import torch_geometric
import os
import pickle
import pandas as pd
import numpy as np
import random
import warnings
from .utils import one_hot_encode, generate_edges, sequence2int_np, computeBPPM

warnings.filterwarnings("ignore", category=FutureWarning)

class SingleRfamDataset(torch_geometric.data.Dataset): 
    
    ###Contains single RFAM family
    
    def __init__(self, rfam_dir, rfam_id, transform=None, pre_transform=None, pre_filter=None):
        self.dir = rfam_dir
        self.id = rfam_id
        super().__init__(rfam_dir, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return(os.path.join(self.dir, self.id, self.id+".npy"))
    
    @property
    def processed_file_names(self):
        if (os.path.exists(os.path.join(self.dir,self.id,"pt"))): 
            return os.listdir(os.path.join(self.dir,self.id,"pt"))
        else:
            return []
    
    
    def process(self):
        if not os.path.exists(os.path.join(self.dir, self.id,"pt")):
            os.makedirs(os.path.join(self.dir, self.id,"pt"))

        if len(os.listdir(os.path.join(self.dir, self.id,"pt"))) == 0 and os.path.exists(self.raw_file_names):
            with open(self.raw_file_names,"rb") as f:
                while True:
                    try: 
                        array = np.load(f, allow_pickle=True)
                        
                        
                        ## seq (node features one-hot encoded sequence) and classes (node classes) are essentially the same right now. Just for proof of concept
                        ## TODO replace with significant features (e.g. position/order in sequence)
                        
                        seq = one_hot_encode(array.item()['seq_int'])
                        if seq is None: ## if sequence contains unknown base 
                            continue
                        seq = torch.tensor(seq)
                        classes = torch.tensor(array.item()['seq_int'])
                        
                        edge_data = generate_edges(len(array.item()['seq_int']),array.item()['bpp'])
                        edges = torch.tensor(edge_data[["A","B"]].to_numpy())
                        edge_attributes = torch.tensor(edge_data["h_bond"].to_numpy())
                        rfam_id = array.item()['id'].replace("/","_")
                        torch.save({"seq":seq, "edges":edges, "edge_weight":edge_attributes, "rfam":array.item()['rfam_id'],"classes":classes ,"id":rfam_id}, os.path.join(self.dir, self.id,"pt",rfam_id+".pt"))

                    except pickle.UnpicklingError:
                        
                        break
            f.close()

        
        
    def __len__(self):
        return len(self.processed_file_names)
    
    def __getitem__(self, index):
        data = torch.load(os.path.join(self.dir,self.id,"pt",self.processed_file_names[index]))
        return torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),edge_weight=data["edge_weight"],rfam=data["rfam"],ID=data["id"])



class CombinedRfamDataset(torch_geometric.data.Dataset):

    def __init__(self, rfam_dir, rfam_ids, new_dataset_id, seq_length_lim=None,transform=None, pre_transform=None, pre_filter=None):
        self.dir = rfam_dir
        self.ids = rfam_ids
        self.seq_length_lim=seq_length_lim
        self.processed_files = []
        
        if new_dataset_id is None:
            new_dataset_id="_".join(rfam_ids if seq_length_lim is None else rfam_ids+[str(seq_length_lim)])

        self.data_id=new_dataset_id

        if not os.path.exists(os.path.join(rfam_dir,new_dataset_id)):
            os.makedirs(os.path.join(rfam_dir,new_dataset_id))

        super().__init__(rfam_dir, transform, pre_transform, pre_filter)
    
    @property
    def processed_file_names(self):
        return self.processed_files

    def process(self):
        if os.path.exists(os.path.join(self.dir,self.data_id,"files.list")):
            with open(os.path.join(self.dir,self.data_id,"files.list"),"r") as files:
                self.processed_files.extend(files.read().splitlines())
        else:
            for rfam in self.ids:
                print(rfam)
                single_familiy_datset = SingleRfamDataset(self.dir,rfam)

                if self.seq_length_lim is not None:
                    for sample in single_familiy_datset.processed_file_names:
                        rna = torch.load(os.path.join(self.dir,rfam,"pt",sample))
                        if (len(rna["seq"])<=self.seq_length_lim):
                            self.processed_file_names.append(os.path.join(rfam,"pt",sample))
                else:
                    self.processed_files.extend([rfam+"/pt/"+sample for sample in single_familiy_datset.processed_file_names])
            
            with open(os.path.join(self.dir,self.data_id,"files.list"), "w") as files_list:
                files_list.write("\n".join(self.processed_files))
        print("Processing complete")

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, index):
        data = torch.load(os.path.join(self.dir,self.processed_file_names[index]))
        return torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),edge_weight=data["edge_weight"],rfam=data["rfam"],ID=data["id"])


class Dataset_UTR5(torch_geometric.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.files = []
        super().__init__(dataset_path)
    
    @property
    def processed_file_names(self):
        return self.files
    
    def process(self):
        self.files = os.listdir(os.path.join(self.dataset_path,"pt"))

    def __len__(self):
        return len(self.processed_file_names)
    
    def __getitem__(self, index):
        data = torch.load(os.path.join(self.dataset_path,"pt",self.files[index]))
        return torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),edge_weight=data["edge_weight"],mrl=data["mrl"],ID=data["id"])

class Dataset_UTR5_hetero(torch_geometric.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.files = []
        super().__init__(dataset_path)
    
    @property
    def processed_file_names(self):
        return self.files
    
    def process(self):
        self.files = os.listdir(os.path.join(self.dataset_path,"pt"))

    def __len__(self):
        return len(self.processed_file_names)
    
    def __getitem__(self, index):
        data = torch.load(os.path.join(self.dataset_path,"pt",self.files[index]))
        hetero_data = torch_geometric.data.HeteroData(x=data["seq"],edge_index={"b_pairs":data["edges"].t().contiguous(), "backbone":data["backbone_edges"]},edge_weight=data["edge_weight"],mrl=data["mrl"],ID=data["id"], library=data["library"], designed=data["designed"])
        return hetero_data




                    
                
