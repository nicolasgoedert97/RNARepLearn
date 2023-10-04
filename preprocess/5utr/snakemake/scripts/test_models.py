from RNARepLearn.models import *
from RNARepLearn.modules import *
from tensorflow.python.summary.summary_iterator import summary_iterator
from sklearn.metrics import r2_score
import configparser
import gin
import io
import numpy as np
import pandas as pd
import os
import torch_geometric
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
from scipy.ndimage import gaussian_filter1d
import torch

gin.external_configurable(torch_geometric.nn.GCNConv, "GCNConv")
gin.external_configurable(torch_geometric.nn.TransformerConv, "TransformerConv")
gin.external_configurable(torch_geometric.transforms.AddLaplacianEigenvectorPE, "AddLaplacianEigenvectorPE")
gin.external_configurable(torch_geometric.nn.ChebConv, "ChebConv")
gin.external_configurable(torch_geometric.nn.GraphConv, "GraphConv")
gin.external_configurable(torch_geometric.nn.GatedGraphConv, "GatedGraphConv")
gin.external_configurable(torch_geometric.nn.TAGConv, "TAGConv")
gin.external_configurable(torch_geometric.nn.GATConv, "GATConv")


class Loss():
    def __init__(self, loss):
        if len(loss)>0:
            self.step, self.value = zip(*loss)
        else:
            self.step, self.value = None, None
            
class Experiment():
    def __init__(self, path):
        self.path=path
        self.runs = {}
        for dir in os.listdir(self.path):
            if dir != "indices":
                self.runs[dir] = Run(os.path.join(self.path, dir))

    def __str__(self):
        return str(list(self.runs.keys()))
            
class Run():
    def __init__(self, path):
        self.path=path
        self.folds = []
        for fold in os.listdir(os.path.join(path)):
            print(path)
            #self.folds.append(Fold(os.path.join(path,fold), load_model=True))
            
        
class Fold():
    def __init__(self, path, load_model=True):
        importlib.reload(gin)
        self.path = path
        self.files = os.listdir(path)
        self.loss, self.val_loss, self.edge_loss, self.edge_val_loss, self.node_loss, self.node_val_loss, self.node_accuracy, self.node_accuracy_val, self.epoch_loss, self.epoch_loss_val = self.get_metrics()
        if os.path.exists(os.path.join(path, "test/pred_results.csv")):
            self.pred_results = pd.read_csv(os.path.join(path, "test/pred_results.csv"))
        
        if load_model:
            self.parse_config_file()
            self.model, self.models = self.load_model(model_selection="model_epoch18")
    
    def load_model(self, model_selection=None):
        models = []
        for i in self.files:
            if "model" in i:
                models.append(i)
        
        print(models)
        model = Encoder_Decoder_Model()
        if model_selection is None:
            
            model.load_state_dict(torch.load(os.path.join(self.path,models[-1])))
        else:
            model.load_state_dict(torch.load(os.path.join(self.path,model_selection)))
        
        return model, models
        
    
    def parse_config_file(self):
        with open(os.path.join(self.path, "gin.config")) as conf:
            filedata = conf.read()
            filedata = filedata.replace("command_line.train.encoder", "models.Encoder_Decoder_Model.encoder")
            filedata = filedata.replace("command_line.train.decoder", "models.Encoder_Decoder_Model.decoder")
            conf = io.StringIO(filedata)
            restore = []
            for f in conf:
                if not f.strip().startswith("command_line.train.pretrain_decoder"):
                    restore.append(f)
            conf = io.StringIO("".join(restore))
            gin.parse_config(conf, skip_unknown=True)                
                
            
    def createDf(train, val, name):
        data = np.array(train).T
        df = pd.DataFrame({name+"_train":data[1]})
        df = df.set_index(data[0])

        data = np.array(val).T
        df2 = pd.DataFrame({name+"_val":data[1]})
        df2 = df2.set_index(data[0])

        return pd.merge(df, df2,how="outer" ,left_index=True, right_index=True)
    
    def get_metrics(self):
        events = None
        for f in self.files:
            if f.startswith("events.out"):
                events = f

        trains = []
        vals = []

        edges_trains = []
        edges_vals = []

        node_trains = []
        node_vals = []

        node_accuracy = []
        node_accurcy_val = []

        epoch_loss = []
        epoch_loss_val = []

        for e in summary_iterator(os.path.join(self.path, events)):
            for v in e.summary.value:
                if v.tag == 'Loss/train':
                    trains.append((e.step,v.simple_value))
                if v.tag == 'Loss/val':
                    vals.append((e.step,v.simple_value))
                if v.tag == 'Loss_edges/train':
                    edges_trains.append((e.step,v.simple_value))
                if v.tag == 'Loss_edges/val':
                    edges_vals.append((e.step,v.simple_value))
                if v.tag == 'Loss_nodes/train':
                    node_trains.append((e.step,v.simple_value))
                if v.tag == 'Loss_nodes/val':
                    node_vals.append((e.step,v.simple_value))
                if v.tag == 'Node_Accuracy/train':
                    node_accuracy.append((e.step,v.simple_value))
                if v.tag == 'Node_Accuracy/val':
                    node_accurcy_val.append((e.step,v.simple_value))
                if v.tag == 'Epoch_Loss/train':
                    epoch_loss.append((e.step,v.simple_value))
                if v.tag == 'Epoch_Loss/val':
                    epoch_loss_val.append((e.step,v.simple_value))
        return [Loss(trains), Loss(vals), Loss(edges_trains), Loss(edges_vals), Loss(node_trains), Loss(node_vals), Loss(node_accuracy), Loss(node_accurcy_val), Loss(epoch_loss), Loss(epoch_loss_val)]
    
    def parse_gin(self):
        pass


run_names = ["runsGCN","runsGCN_3layers","runsGCN_6layers","runsRPI","runsRPI_3layers","runsRPI_6layers","runsGatedGraphConv","runsGatedGraphConv_3layers","runsGatedGraphConv_6layers", "runsGCN_9layers_LSTM","runsGCN_3layers_LSTM","runsGCN_6layers_LSTM"]
run_path = "/p/project/hai_rnareplearn/RUNS/600k_pretraining_25epochs"

Models = []
for i in run_names:
    path = os.path.join(run_path,i,"fold0")
    run = Fold(path, load_model=True)
    Models.append(run)

from RNARepLearn.datasets import GFileDataset
rfam = GFileDataset("/p/project/hai_rnareplearn/under_300",["u300"])

from RNARepLearn.utils import indexed_train_val_test_loaders, mask_batch, reconstruct_bpp
from tqdm import tqdm
train, val, test = indexed_train_val_test_loaders(rfam, os.path.join(run_path,"indices/folds/fold0/train.indices"),os.path.join(run_path,"indices/folds/fold0/val.indices"), os.path.join(run_path,"indices/test.indices"), 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cel_loss = torch.nn.CrossEntropyLoss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')



mask = False



with torch.no_grad():
    for i,model in enumerate(Models):
        print(run_names[i])
        
        if os.path.exists(os.path.join(run_path,run_names[i],"test/test.csv")):
            print(os.path.join(run_path,run_names[i],"test/test.csv")+" exists! Skipping")
            continue
        
        os.makedirs(os.path.join(run_path,run_names[i],"test"), exist_ok=True)
        losses={"RFAM":[], "node_loss":[], "edge_loss":[]}
        
        n=0
        model.model.to(device)
        for batch in tqdm(test):
            n+=1
            true_x = torch.clone(batch.x[:,:4])#[:,:4]
            true_weights = torch.clone(batch.edge_weight)

            if mask:
                train_mask = mask_batch(batch,15)

            batch.to(device)

            nucs, bpp = model.model(batch)

            if mask:
                node_loss = cel_loss(nucs.cpu()[train_mask],true_x[train_mask])
                edge_loss = kl_loss(bpp[train_mask].cpu(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0]))))
            else:
                node_loss = cel_loss(nucs.cpu(),true_x)
                edge_loss = kl_loss(bpp.cpu(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0]))) )

            losses["RFAM"].append(batch.rfam[0])
            losses["node_loss"].append(node_loss.numpy())
            losses["edge_loss"].append(edge_loss.numpy())
            
        df = pd.DataFrame(losses)
        df = df.astype({'node_loss':'float','edge_loss':'float'})
        df.to_csv(os.path.join(run_path,run_names[i],"test/test.csv"))