import torch
import torch_geometric
import os
import pickle
import pandas as pd
import numpy as np
import random
import gin
import warnings
from bioio.tf import GFileTFRecord
from .utils import one_hot_encode, generate_edges, sequence2int_np, add_backbone_single
from torch_geometric.transforms import AddLaplacianEigenvectorPE

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
        return os.listdir(os.path.join(self.dir, self.id, "pt"))
    
    
    def process(self):
        if not os.path.exists(os.path.join(self.dir, self.id,"pt")):
            os.makedirs(os.path.join(self.dir, self.id,"pt"))

        if len(os.listdir(os.path.join(self.dir, self.id,"pt"))) == 0 and os.path.exists(self.raw_file_names):
            # with open(self.raw_file_names,"rb") as f:
            #     while True:
            #         try: 
            #             array = np.load(f, allow_pickle=True)
                        
                        
            #             ## seq (node features one-hot encoded sequence) and classes (node classes) are essentially the same right now. Just for proof of concept
            #             ## TODO replace with significant features (e.g. position/order in sequence)
                        
            #             seq = one_hot_encode(array.item()['seq_int'])
            #             if seq is None: ## if sequence contains unknown base 
            #                 continue
            #             seq = torch.tensor(seq)
            #             classes = torch.tensor(array.item()['seq_int'])
                        
            #             edge_data = generate_edges(len(array.item()['seq_int']),array.item()['bpp'])
            #             edges = torch.tensor(edge_data[["A","B"]].to_numpy())
            #             edge_attributes = torch.tensor(edge_data["h_bond"].to_numpy())
            #             rfam_id = array.item()['id'].replace("/","_")
            #             torch.save({"seq":seq, "edges":edges, "edge_weight":edge_attributes, "rfam":array.item()['rfam_id'],"classes":classes ,"id":rfam_id}, os.path.join(self.dir, self.id,"pt",rfam_id+".pt"))

            #         except pickle.UnpicklingError:
                        
            #             break
            # f.close()
            print(rfam, "empty!!!")

        
        
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, index):

        data = torch.load(os.path.join(self.dir,self.id,"pt",self.processed_file_names[index]))
        return torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),edge_weight=data["edge_weight"],rfam=data["rfam"],ID=data["id"])


@gin.configurable
class CombinedRfamDataset(torch_geometric.data.Dataset):

    def __init__(self, rfam_dir, rfam_ids, new_dataset_id, seq_length_lim=None,transform=None, pre_transform=None, pre_filter=None, sample_rfams=None, add_backbone=False, add_LPE=False):
        self.dir = rfam_dir
        self.ids = rfam_ids
        self.seq_length_lim=seq_length_lim
        self.processed_files = []
        self.sample_rfams = sample_rfams
        self.add_bb = add_backbone
        self.add_LPE = add_LPE
        
        if new_dataset_id is None:
            new_dataset_id="_".join(rfam_ids if seq_length_lim is None else rfam_ids+[str(seq_length_lim)])

        self.data_id=new_dataset_id

        if not os.path.exists(os.path.join(rfam_dir,new_dataset_id)):
            os.makedirs(os.path.join(rfam_dir,new_dataset_id))

        super().__init__(rfam_dir, transform, pre_transform, pre_filter)
    
    def pre_compute_Laplacian(self,k):
        pe = AddLaplacianEigenvectorPE(k)
        

        for i,f in enumerate(self.processed_file_names):
            if (i%100 == 0):
                print(str(i)+"/"+str(len(self.processed_file_names)))
            if not os.path.exists(os.path.join(self.dir,f.split("/")[0],"laplacianPE_"+str(k))):
                os.makedirs(os.path.join(self.dir,f.split("/")[0],"laplacianPE_"+str(k)))
            data = torch.load(os.path.join(self.dir,f))
            edge_index, edge_weight, edge_attr = add_backbone_single(data["seq"], data["edges"].t().contiguous(), data["edge_weight"])
            data = torch_geometric.data.Data(x=data["seq"],edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, rfam=data["rfam"], ID=data["id"])
            data = pe(data)
            torch.save(data.laplacian_eigenvector_pe, os.path.join(self.dir,f.split("/")[0],"laplacianPE_"+str(k), f.split("/")[-1]))
            
            
    
    @property
    def processed_file_names(self):
        return self.processed_files

    def process(self):
        if os.path.exists(os.path.join(self.dir,self.data_id,"files.list")):
            with open(os.path.join(self.dir,self.data_id,"files.list"),"r") as files:
                if self.sample_rfams is None:
                    self.processed_files.extend(files.read().splitlines())
                else:
                    rfams = files.read().splitlines()
                    for rfam in rfams:
                        rfam_name = rfam[:7]

                        if rfam_name in self.sample_rfams:
                            self.processed_files.append(rfam)

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
        if self.add_bb:
            edge_index, edge_weight, edge_attr = add_backbone_single(data["seq"], data["edges"].t().contiguous(), data["edge_weight"])
            data = torch_geometric.data.Data(x=data["seq"],edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, rfam=data["rfam"], ID=data["id"])
        else:
            data = torch_geometric.data.Data(x=data["seq"],edge_index=data["edges"].t().contiguous(),edge_weight=data["edge_weight"],rfam=data["rfam"],ID=data["id"])
        
        if self.add_LPE:
            data.x = torch.cat((data.x, torch.load(os.path.join(self.dir,self.processed_file_names[index].split("/")[0],"laplacianPE_"+str(16), self.processed_file_names[index].split("/")[-1]))), dim=1)
        if self.transform is not None:
            data = self.transform(data)
        return data
            



class Dataset_UTR5(torch_geometric.data.Dataset):
    def __init__(self, dataset_path, mask=False):
        self.dataset_path = dataset_path
        self.files = []
        self.mask = mask
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



class GFileDataset(torch_geometric.data.Dataset):
    def __init__(self, base, rfams_list):
        self.rfams = {}
        self.rfam_index_offset = []
        self.len = 0

        for rfam in rfams_list:
            tfr_path = os.path.join(base, rfam, "tfrecord", rfam+".tfrecord")
            if os.path.exists(tfr_path):
                tfr = GFileTFRecord(tfr_path, features=tfr_path+".features.json", index=os.path.join(base, rfam, "tfrecord", rfam)+".index", to_numpy=True)
                self.rfams[rfam] = tfr
                new_len = self.len + len(tfr)
                self.rfam_index_offset.append([self.len, new_len])
                self.len = new_len
    
    def fetch_by_id(self, rfam, id):
        rfam_tfr = self.rfams[rfam]
        index_offset = self.rfam_index_offset[list(self.rfams.keys()).index(rfam)]

        for i, sample in enumerate(rfam_tfr):
            if sample['id'].decode("utf-8")==id:
                return i+index_offset[0]
        
        return None
        #for idx in self.rfam_index_offset


    def _find_matching_rfam(self, idx):
        for i, index_range in enumerate(self.rfam_index_offset):
            if idx >= index_range[0] and idx<index_range[1]:
                return list(self.rfams.keys())[i], index_range

        raise ValueError("No Rfam Range found!")
    
    def print_ranges(self):
        for i,key in enumerate(list(self.rfams.keys())):
            print(key+"\t["+str(self.rfam_index_offset[i][0])+","+str(self.rfam_index_offset[i][1])+"]")

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return self.len
    
    def get(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        rfam , index_range = self._find_matching_rfam(index)
        rfam_index = index - index_range[0]

        np_element = self.rfams[rfam][rfam_index]

        X = torch.tensor(np_element["seq"])
        EDGE_INDEX = torch.tensor(np_element["edges"]).t().contiguous()
        EDGE_WEIGHT = torch.tensor(np_element["edge_weight"])
        return torch_geometric.data.Data(x=X,edge_index=EDGE_INDEX,edge_weight=EDGE_WEIGHT, rfam=np_element["rfam"].decode("utf-8"),ID=np_element["id"].decode("utf-8")) #
                    
                
class GFileDatasetUTR(torch_geometric.data.Dataset):
    def __init__(self, base, rfams_list):
        self.rfams = {}
        self.rfam_index_offset = []
        self.len = 0

        for rfam in rfams_list:
            tfr_path = os.path.join(base, rfam, "tfrecord", rfam+".tfrecord")
            if os.path.exists(tfr_path):
                tfr = GFileTFRecord(tfr_path, features=tfr_path+".features.json", index=os.path.join(base, rfam, "tfrecord", rfam)+".index", to_numpy=True)
                self.rfams[rfam] = tfr
                new_len = self.len + len(tfr)
                self.rfam_index_offset.append([self.len, new_len])
                self.len = new_len
    
    def fetch_by_id(self, rfam, id):
        rfam_tfr = self.rfams[rfam]
        index_offset = self.rfam_index_offset[list(self.rfams.keys()).index(rfam)]

        for i, sample in enumerate(rfam_tfr):
            if sample['id'].decode("utf-8")==id:
                return i+index_offset[0]
        
        return None
        #for idx in self.rfam_index_offset


    def _find_matching_rfam(self, idx):
        for i, index_range in enumerate(self.rfam_index_offset):
            if idx >= index_range[0] and idx<index_range[1]:
                return list(self.rfams.keys())[i], index_range

        raise ValueError("No Rfam Range found!")
    
    def print_ranges(self):
        for i,key in enumerate(list(self.rfams.keys())):
            print(key+"\t["+str(self.rfam_index_offset[i][0])+","+str(self.rfam_index_offset[i][1])+"]")

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return self.len
    
    def get(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        rfam , index_range = self._find_matching_rfam(index)
        rfam_index = index - index_range[0]

        np_element = self.rfams[rfam][rfam_index]

        X = torch.tensor(np_element["seq"])
        EDGE_INDEX = torch.tensor(np_element["edges"]).t().contiguous()
        EDGE_WEIGHT = torch.tensor(np_element["edge_weight"])
        return torch_geometric.data.Data(x=X,edge_index=EDGE_INDEX,edge_weight=EDGE_WEIGHT, mrl=torch.tensor(np_element["rl"]), ID=np_element["id"]) #
                    
                
class GFileDatasetAffinity(torch_geometric.data.Dataset):
    def __init__(self, base, rfams_list):
        self.rfams = {}
        self.rfam_index_offset = []
        self.len = 0

        for rfam in rfams_list:
            tfr_path = os.path.join(base, rfam, "tfrecord", rfam+".tfrecord")
            if os.path.exists(tfr_path):
                tfr = GFileTFRecord(tfr_path, features=tfr_path+".features.json", index=os.path.join(base, rfam, "tfrecord", rfam)+".index", to_numpy=True)
                self.rfams[rfam] = tfr
                new_len = self.len + len(tfr)
                self.rfam_index_offset.append([self.len, new_len])
                self.len = new_len
    
    def fetch_by_id(self, rfam, id):
        rfam_tfr = self.rfams[rfam]
        index_offset = self.rfam_index_offset[list(self.rfams.keys()).index(rfam)]

        for i, sample in enumerate(rfam_tfr):
            if sample['id'].decode("utf-8")==id:
                return i+index_offset[0]
        
        return None
        #for idx in self.rfam_index_offset


    def _find_matching_rfam(self, idx):
        for i, index_range in enumerate(self.rfam_index_offset):
            if idx >= index_range[0] and idx<index_range[1]:
                return list(self.rfams.keys())[i], index_range

        raise ValueError("No Rfam Range found!")
    
    def print_ranges(self):
        for i,key in enumerate(list(self.rfams.keys())):
            print(key+"\t["+str(self.rfam_index_offset[i][0])+","+str(self.rfam_index_offset[i][1])+"]")

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return self.len
    
    def get(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        rfam , index_range = self._find_matching_rfam(index)
        rfam_index = index - index_range[0]

        np_element = self.rfams[rfam][rfam_index]

        X = torch.tensor(np_element["seq"])
        EDGE_INDEX = torch.tensor(np_element["edges"]).t().contiguous()
        EDGE_WEIGHT = torch.tensor(np_element["edge_weight"])
        return torch_geometric.data.Data(x=X,edge_index=EDGE_INDEX,edge_weight=EDGE_WEIGHT, binding_prots=torch.tensor(np_element["binding_prots"]), ID=np_element["id"]) #
  

class GFileDatasetAffinity2(torch_geometric.data.Dataset):
    def __init__(self, base, rfams_list):
        print(rfams_list)
        self.rfams = {}
        self.rfam_index_offset = []
        self.len = 0

        for rfam in rfams_list:
            tfr_path = os.path.join(base, rfam, "tfrecord", rfam+".tfrecord")
            if os.path.exists(tfr_path):
                tfr = GFileTFRecord(tfr_path, features=tfr_path+".features.json", index=os.path.join(base, rfam, "tfrecord", rfam)+".index", to_numpy=True)
                self.rfams[rfam] = tfr
                new_len = self.len + len(tfr)
                self.rfam_index_offset.append([self.len, new_len])
                self.len = new_len

    def fetch_by_id(self, rfam, id):
        rfam_tfr = self.rfams[rfam]
        index_offset = self.rfam_index_offset[list(self.rfams.keys()).index(rfam)]

        for i, sample in enumerate(rfam_tfr):
            if sample['id'].decode("utf-8")==id:
                return i+index_offset[0]
        
        return None
        #for idx in self.rfam_index_offset


    def _find_matching_rfam(self, idx):
        for i, index_range in enumerate(self.rfam_index_offset):
            if idx >= index_range[0] and idx<index_range[1]:
                return list(self.rfams.keys())[i], index_range

        raise ValueError("No Rfam Range found!")

    def print_ranges(self):
        for i,key in enumerate(list(self.rfams.keys())):
            print(key+"\t["+str(self.rfam_index_offset[i][0])+","+str(self.rfam_index_offset[i][1])+"]")

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.len

    def get(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        rfam , index_range = self._find_matching_rfam(index)
        rfam_index = index - index_range[0]

        np_element = self.rfams[rfam][rfam_index]

        X = torch.tensor(np_element["x"])
        EDGE_INDEX = torch.tensor(np_element["edge_index"])
        EDGE_WEIGHT = torch.tensor(np_element["edge_weight"])
        return torch_geometric.data.Data(x=X,edge_index=EDGE_INDEX,edge_weight=EDGE_WEIGHT, binding_prots=torch.tensor(np_element["binding_prots"]), ID=torch.argmax(torch.tensor(np_element["binding_prots"])) ) #
