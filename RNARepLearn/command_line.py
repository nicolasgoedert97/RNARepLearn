import RNARepLearn
import argparse
import gin
import torch_geometric
import numpy as np
import shutil
import os
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from RNARepLearn.modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder
from RNARepLearn.datasets import CombinedRfamDataset, Dataset_UTR5
from RNARepLearn.utils import train_val_test_loaders, save_dataset, generate_edges, k_fold_loaders, reconstruct_bpp
from RNARepLearn.models import Encoder_Decoder_Model
from RNARepLearn.train import MaskedTraining, AutoEncoder, TETraining



def main():

    gin.external_configurable(torch_geometric.nn.GCNConv, "GCNConv")
    gin.external_configurable(torch_geometric.nn.TransformerConv, "TransformerConv")
    gin.external_configurable(torch_geometric.transforms.AddLaplacianEigenvectorPE, "AddLaplacianEigenvectorPE")
    gin.external_configurable(torch_geometric.nn.ChebConv, "ChebConv")
    gin.external_configurable(torch_geometric.nn.GraphConv, "GraphConv")
    gin.external_configurable(torch_geometric.nn.GatedGraphConv, "GatedGraphConv")
    gin.external_configurable(torch_geometric.nn.TAGConv, "TAGConv")
    gin.external_configurable(torch_geometric.nn.GATConv, "GATConv")


    parser = argparse.ArgumentParser()
    parser.add_argument('--gin')
    parser.add_argument('--output')
    parser.add_argument("--test-on-train", action="store_true")

    args = parser.parse_args()


    gin.parse_config_file(args.gin)

    train_loader, val_loader, test_loader, batch_size = setup_data()

    logdirs = []
    models = []
    if isinstance(train_loader, list):
        for i in range(len(train_loader)):
            print("Fold "+str(i))
            model, log_dir, train_device, train_mode = train(train_loader=train_loader[i], val_loader=val_loader[i], batch_size=batch_size, args=args, log_dir=os.path.join(args.output,"fold"+str(i)))
            if not os.path.exists(os.path.join(log_dir,"datasets")):
                os.makedirs(os.path.join(log_dir,"datasets"))
            save_dataset(train_loader.dataset.indices, os.path.join(log_dir,"datasets","train"))
            save_dataset(test_loader.dataset.indices, os.path.join(log_dir,"datasets","test"))
            save_dataset(val_loader.dataset.indices, os.path.join(log_dir,"datasets","val"))
            models.append(model)
            logdirs.append(log_dir)

    else:
        model, log_dir, train_device, train_mode = train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, args=args, log_dir=args.output)
        models.append(model)
        if not os.path.exists(os.path.join(log_dir,"datasets")):
            os.makedirs(os.path.join(log_dir,"datasets"))
        save_dataset(train_loader.dataset.indices, os.path.join(log_dir,"datasets","train"))
        save_dataset(test_loader.dataset.indices, os.path.join(log_dir,"datasets","test"))
        save_dataset(val_loader.dataset.indices, os.path.join(log_dir,"datasets","val"))

    
    test(test_loader, models, logdirs, train_device, train_mode, "test")

    if args.test_on_train:
        test(train_loader, model, log_dir, train_device, train_mode, "train")


@gin.configurable
def setup_data(basepath, train_val_test_splits, batch_size, dataset_type,rfams=None, seq_length_lim=None, dataset_name=None, transform=None, LapPE_k = 60, k_fold=None):
    dataset = None

    print("Dataset name: "+str(dataset_name))

    match dataset_type:
        case "RFAM":
            if transform is not None:
                dataset = CombinedRfamDataset(basepath, rfams, dataset_name, seq_length_lim, transform=transform(LapPE_k))
            else:
                dataset = CombinedRfamDataset(basepath, rfams, dataset_name, seq_length_lim)
        case "UTR":
            dataset = Dataset_UTR5(basepath)
        case None:
            raise Exception("No application mode defined")
        
    print("Dataset length: "+str(len(dataset)))


    if k_fold is not None:
        train, val, test = k_fold_loaders(k_fold, dataset)
    else:
        train, val, test = train_val_test_loaders(dataset, train_val_test_splits[0], train_val_test_splits[1], train_val_test_splits[2], batch_size)

    if len(val) == 0:
        val=None

    return train, val, test, batch_size




@gin.configurable
def train(encoder, decoder, n_epochs, train_loader, batch_size, mode, masked_percentage=15 ,pretrain_decoder = None,model_state_dict = None, log_dir=None, val_loader=None, args=None):

    writer = SummaryWriter(log_dir)

    if args is not None:
        shutil.copyfile(args.gin ,os.path.join(writer.log_dir, "gin.config"))

    if model_state_dict is None:
        model = Encoder_Decoder_Model(encoder=encoder, decoder=decoder)
    
    else:
        #If model is set, model pretrained with SeqStruc decoder
        print("Loading pretrained model")
        pretrained_model = Encoder_Decoder_Model(encoder=encoder, decoder=pretrain_decoder)
        print(pretrained_model)
        pretrained_model.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        encoder = next(pretrained_model.model.children())

        encoder.requires_grad = False

        model = Encoder_Decoder_Model(encoder=encoder, decoder=decoder)


    match mode:
        case "masked":
            training = MaskedTraining(model, n_epochs, masked_percentage, writer)

        case "autoencode":
            training = AutoEncoder(model, n_epochs, writer)

        case "TE":
            training = TETraining(model, n_epochs, writer)

    


    training.run(train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(writer.log_dir, "model"))

    return model, log_dir, training.device, mode


def test(test_loader, models, log_dirs, device, test_mode, test_name):
    for i in range(len(models)):
        print("Testing model "+str(i)+"/"+str(len(models)))
        model = model[i]
        log_dir = log_dirs[i]

        if not os.path.exists(os.path.join(log_dir, "test")):
            os.makedirs(os.path.join(log_dir, "test"))
        model.eval()
        model.to(device)

        for i, batch in enumerate(test_loader[:50]):
            batch.to(device)
            if test_mode == "TE":
                pred_MRL = model(batch)
                df = pd.DataFrame({'pred_mrl':pred_MRL.detach().cpu().numpy(), 'true_mrl':batch.mrl.detach().cpu().numpy(), 'id':batch.ID.cpu()})
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




        
    




    