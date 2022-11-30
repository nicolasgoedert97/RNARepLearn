import torch
import torch_geometric
import os
import pickle
import pandas as pd
import numpy as np

class SingleMaskedRfamDataset(torch_geometric.data.Dataset): 
    
    ###Contains single RFAM family with masked sequence
    
    def __init__(self, rfam_dir, rfam_id, transform=None, pre_transform=None, pre_filter=None):
        self.dir = rfam_dir
        self.id = rfam_id
        super().__init__(rfam_dir, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return(os.path.join(self.dir, self.id, self.id+".npy"))
    
    @property
    def processed_file_names(self):
        return os.listdir(os.path.join(self.dir,self.id,"pt"))
    
    
    def one_hot_encode(self, seq):
        nuc_d = {0:[1.0,0.0,0.0,0.0],
                 1:[0.0,1.0,0.0,0.0],
                 2:[0.0,0.0,1.0,0.0],
                 3:[0.0,0.0,0.0,1.0]}
        vec=np.array([nuc_d[x] for x in seq])
        return vec
    
    def generate_edges(self,seq_len,bpp):
        X = np.zeros((seq_len,seq_len))
        X[np.triu_indices(X.shape[0], k = 1)] = bpp
        X = X+X.T
        df = pd.DataFrame(X)
        np.fill_diagonal(df.values, np.nan)
        adf = df.stack().reset_index()
        adf = adf.rename(columns={"level_0":"A","level_1":"B",0:"weight"})
        return (adf.loc[adf["weight"]!=0.0])[["A","B"]].to_numpy()
    
    
    def process(self):
        if not os.path.exists(os.path.join(self.dir, self.id,"pt")):
            os.makedirs(os.path.join(self.dir, self.id,"pt"))

        if len(os.listdir(os.path.join(self.dir, self.id,"pt"))) == 0:
            with open(self.raw_file_names,"rb") as f:
                while True:
                    try: 
                        array = np.load(f, allow_pickle=True)
                        
                        
                        ## seq (node features one-hot encoded sequence) and classes (node classes) are essentially the same right now. Just for proof of concept
                        ## TODO replace with significant features (e.g. position/order in sequence)
                        
                        seq = torch.tensor(self.one_hot_encode(array.item()['seq_int']))
                        classes = torch.tensor(array.item()['seq_int'])
                        
                        struc = torch.tensor(self.generate_edges(len(array.item()['seq_int']),array.item()['bpp']))
                        rfam_id = array.item()['id'].replace("/","_")
                        torch.save({"seq":seq, "edges":struc, "rfam":array.item()['rfam_id'],"classes":classes ,"id":rfam_id}, os.path.join(self.dir, self.id,"pt",rfam_id+".pt"))
                        
                    except pickle.UnpicklingError:
                        
                        break
            f.close()
        
        
        
    def __len__(self):
        return len(self.processed_file_names)
    
    def __getitem__(self, index):
        self.processed_file_names[index]
        data = torch.load(os.path.join(self.dir,self.id,"pt",self.processed_file_names[index]))
        return torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),y=data["classes"],rfam=data["rfam"],ID=data["id"])