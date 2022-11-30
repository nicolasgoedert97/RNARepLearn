import torch
import torch_geometric

class SingleRfamDataset(torch.utils.data.Dataset): 
    
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

    
        
        
        
    
    def __init__(self, rfam_dir, rfam_id, n_samples):
        self.data = np.load(open(os.path.join(rfam_dir, rfam_id,rfam_id+".npz"),"rb"),allow_pickle=True)
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        seq = self.one_hot_encode(list(self.data["sequences"][index]))
        struc = self.generate_edges(len(seq),self.data["structures"][index])
        rfam = self.data["ids"][index].split("/")[0]
        rfam_id = self.data["ids"][index].split("/")[1]+ "/" +self.data["ids"][index].split("/")[2]
        
        return torch.tensor(seq), torch.tensor(struc), rfam, rfam_id