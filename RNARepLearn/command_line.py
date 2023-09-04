import os
import RNARepLearn
import argparse
import gin
import torch_geometric
import numpy as np
import shutil
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from RNARepLearn.modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder, Affinity_Decoder
from RNARepLearn.datasets import CombinedRfamDataset, Dataset_UTR5, GFileDataset, GFileDatasetUTR, GFileDatasetAffinity, GFileDatasetAffinity2
from RNARepLearn.utils import random_train_val_test_loaders, indexed_train_val_test_loaders, save_dataset, generate_edges, reconstruct_bpp
from RNARepLearn.models import Encoder_Decoder_Model
from RNARepLearn.train import MaskedTraining, AutoEncoder, TETraining, AffinityTraining
import datetime




def main():
    
    ## External configs for gin
    gin.external_configurable(torch_geometric.nn.GCNConv, "GCNConv")
    gin.external_configurable(torch_geometric.nn.TransformerConv, "TransformerConv")
    gin.external_configurable(torch_geometric.transforms.AddLaplacianEigenvectorPE, "AddLaplacianEigenvectorPE")
    gin.external_configurable(torch_geometric.nn.ChebConv, "ChebConv")
    gin.external_configurable(torch_geometric.nn.GraphConv, "GraphConv")
    gin.external_configurable(torch_geometric.nn.GatedGraphConv, "GatedGraphConv")
    gin.external_configurable(torch_geometric.nn.TAGConv, "TAGConv")
    gin.external_configurable(torch_geometric.nn.GATConv, "GATConv")


    ##Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_type', required=True)

    # optional arguments
    parser.add_argument("--test-on-train", action="store_true")
    parser.add_argument('--dataset_names', nargs='+')
    parser.add_argument('--write_datasets', action="store_true")


    parser.add_argument('--train_indices')
    parser.add_argument('--val_indices')
    parser.add_argument('--test_indices')

    parser.add_argument('--eval_model', action="store_true")

    parser.add_argument('--train_split')
    parser.add_argument('--val_split')
    parser.add_argument('--test_split')
    parser.add_argument('--data_parallel', action="store_true")

    parser.add_argument('--train_mode')
    parser.add_argument('--model_state_dict')
    parser.add_argument('--lr')

    args = parser.parse_args()
    gin.parse_config_file(args.gin)

    ## Checking if Train, Val and Test sets are given.
    splits = None
    if args.train_indices is None:
        print("No indices given. Using dataset splits.")
        if args.train_split is None or args.val_split is None or args.test_split is None:
            print("Not all dataset splits given. Using default.")
            splits=[0.8,0.1,0.1]
        else:
            splits=[float(args.train_split),float(args.val_split),float(args.test_split)]
            assert sum(splits) == 1.0, "Splits not summing to 1!"
    else:
        indices = [args.train_indices]
        if args.val_indices is not None:
            indices.append(args.val_indices)
        else:
            indices.append(None)
        if args.test_indices is not None:
            indices.append(args.val_indices)
        else:
            indices.append(None)
    print("DEBUG")
    print(print(args))
    
    ## Setting up data
    train_loader, val_loader, test_loader, batch_size = setup(basepath=args.dataset_path, dataset_names=args.dataset_names, dataset_type=args.dataset_type, train_val_test_splits=splits, indices=indices, parallel=args.data_parallel)

    print(next(iter(train_loader)))
    
    model, log_dir, train_device, train_mode = train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, args=args, log_dir=args.output, data_parallel=args.data_parallel, model_state_dict=args.model_state_dict, mode=args.train_mode)
    if args.eval_model:
        test(val_loader, model, log_dir, args.train_mode, "results")
    

    shutil.copyfile(args.gin, os.path.join(log_dir, "gin.config"))




@gin.configurable
def setup(basepath, dataset_names, dataset_type, batch_size, train_val_test_splits=None, indices=None, parallel=False):

    print("Datasetnames "+" ".join(dataset_names))
    if dataset_names is None:
        dataset_names = os.listdir(basepath)

    match dataset_type:
        case "UTR":
            dataset = GFileDatasetUTR(basepath, dataset_names)
        case "RFAM":
            dataset = GFileDataset(basepath, dataset_names)
        case "Affinity":
            dataset_names = ["150_rbps"]
            dataset = GFileDatasetAffinity(basepath, dataset_names)
        case _:
            raise Exception("No valid application mode defined")
    
    print("Dataset length: "+str(len(dataset)))

    if train_val_test_splits is not None:
        train, val, test = random_train_val_test_loaders(dataset, train_val_test_splits[0], train_val_test_splits[1], train_val_test_splits[2], batch_size)

    if indices is not None:
        print("num_workers: "+str(int(os.cpu_count()/4)/2))

        train, val, test = indexed_train_val_test_loaders(dataset, indices[0], indices[1], indices[2], batch_size, num_workers=int(os.cpu_count()/150), parallel=parallel)

    return train, val, test, batch_size



@gin.configurable
def train(n_epochs, train_loader, batch_size, mode, data_parallel=False ,masked_percentage=15 ,model_state_dict = None, log_dir=None, val_loader=None, args=None, fixed_encoder=False, double_output_channels=False):

    writer = SummaryWriter(log_dir)

    model = Encoder_Decoder_Model()
    
    if model_state_dict is not None:
        print("Loading pretrained model")
        model.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    
    encoder = next(model.model.children())
    if fixed_encoder:
        encoder.requires_grad = False

    if mode == "TE":
        encoder_outputchannels = encoder.output_channels

        if isinstance(encoder, RPINetEncoder) or double_output_channels:
            encoder_outputchannels = encoder_outputchannels*2
            
        model = Encoder_Decoder_Model(encoder=encoder, decoder=TE_Decoder(batch_size, encoder_outputchannels))
    
    if mode == "Affinity":
        encoder_outputchannels = encoder.output_channels
        if isinstance(encoder, RPINetEncoder) or double_output_channels:
            encoder_outputchannels = encoder_outputchannels*2
        model = Encoder_Decoder_Model(encoder=encoder, decoder=Affinity_Decoder(encoder_outputchannels, batch_size, 150))
        
    
    if data_parallel:
            print("Using DataParallel")
            model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    match mode:
        case "masked":
            training = MaskedTraining(model, n_epochs, masked_percentage, writer, parallel=data_parallel)

        case "autoencode":
            training = AutoEncoder(model, n_epochs, writer)

        case "TE":
            training = TETraining(model, n_epochs, writer)
        
        case "Affinity":
            training = AffinityTraining(model, n_epochs, writer)

    if mode == "Affinity":

        training.run(train_loader, 150, val_loader)
    else:
        training.run(train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(writer.log_dir, "final_model"))

    return model, log_dir, training.device, mode


def test(test_loader, model ,log_dir, test_mode, test_name):
    
    print("Testing model ")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(os.path.join(log_dir, "test")):
        os.makedirs(os.path.join(log_dir, "test"))
    model.eval()
    model.to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch.to(device)

            if test_mode == "Affinity":

                
                pred_affinity = model(batch)

                df = pd.DataFrame({'pred_mrl':torch.flatten(pred_affinity).detach().cpu().numpy(), 'true_mrl':batch.binding_prots.detach().cpu().numpy()})
                df.to_csv(os.path.join(log_dir, "test", "pred_"+test_name+".csv"), mode='a', header=(i==0), index=False)


            if test_mode == "TE":
                pred_MRL = model(batch)
                df = pd.DataFrame({'pred_mrl':pred_MRL.detach().cpu().numpy(), 'true_mrl':batch.mrl.detach().cpu().numpy(), 'id':batch.ID})
                df.to_csv(os.path.join(log_dir, "test", "pred_"+test_name+".csv"), mode='a', header=(i==0), index=False)

            if test_mode == "masked":
                
                true_x = torch.clone(batch.x)

                pred_nucs, pred_bpp = model(batch)
                true_bpp = torch.tensor(reconstruct_bpp(batch.edge_index, batch.edge_weight, (len(batch.x),len(batch.x))))
                true_mat = true_bpp!=0
                pred_mat = pred_bpp!=0
                all_mat = torch.ones(len(true_mat)**2, dtype=torch.bool).reshape(len(true_mat),len(true_mat))

                TP = torch.sum(torch.logical_and(true_mat, pred_mat)) #intersection true and pred
                FP = torch.sum(torch.logical_and(torch.logical_and(all_mat, ~true_mat), pred_mat)) #setdiff all_edges / true_edges (e.g. all_edges and !ture_edges)
                FN = torch.sum(torch.logical_and(torch.logical_and(all_mat, ~pred_mat), true_mat))


                true_x = true_x.argmax(dim=1)
                pred_x = pred_nucs.argmax(dim=1)

                tp = 0
                fp = 0
                tn = 0
                fn = 0

                for i in range(len(pred_x)): 
                    if true_x[i]==pred_x[i]==1:
                        tp += 1
                    if pred_x[i]==1 and true_x[i]!=pred_x[i]:
                        fp += 1
                    if true_x[i]==pred_x[i]==0:
                        tn += 1
                    if pred_x[i]==0 and true_x[i]!=pred_x[i]:
                        fn += 1
                

                df = pd.DataFrame({"TP":TP, "FP":FP, "FN":FN, "TP_nodes":tp, "FP_nodes":fp, "FN_nodes":fn })
                df.to_csv(os.path.join(log_dir, "test", "pred_"+test_name+".csv"), mode='a', header=(i==0), index=False)
        




        
    




    