import RNARepLearn
import argparse
import gin
import shutil
import os
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from RNARepLearn.modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder
from RNARepLearn.datasets import CombinedRfamDataset, Dataset_UTR5
from RNARepLearn.utils import train_val_test_loaders, save_dataset
from RNARepLearn.models import Encoder_Decoder_Model
from RNARepLearn.train import MaskedTraining, AutoEncoder, TETraining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin')
    parser.add_argument('--output')
    parser.add_argument("--test-on-train", action="store_true")

    args = parser.parse_args()


    gin.parse_config_file(args.gin)

    train_loader, val_loader, test_loader, batch_size = setup_data()

    model, log_dir, train_device, train_mode = train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, args=args, log_dir=args.output)

    if not os.path.exists(os.path.join(log_dir,"datasets")):
        os.makedirs(os.path.join(log_dir,"datasets"))
    save_dataset(train_loader.dataset.indices, os.path.join(log_dir,"datasets","train"))
    save_dataset(test_loader.dataset.indices, os.path.join(log_dir,"datasets","test"))
    save_dataset(val_loader.dataset.indices, os.path.join(log_dir,"datasets","val"))

    
    test(test_loader, model, log_dir, train_device, train_mode, "test")

    if args.test_on_train:
        test(train_loader, model, log_dir, train_device, train_mode, "train")


@gin.configurable
def setup_data(basepath, train_val_test_splits, batch_size, dataset_type,rfams=None, seq_length_lim=None, dataset_name=None):
    dataset = None

    print("Dataset name: "+str(dataset_name))

    match dataset_type:
        case "RFAM":
            dataset = CombinedRfamDataset(basepath, rfams, dataset_name, seq_length_lim)
        case "UTR":
            dataset = Dataset_UTR5(basepath)
        case None:
            raise Exception("No application mode defined")
        
    print("Dataset length: "+str(len(dataset)))

    train, val, test = train_val_test_loaders(dataset, train_val_test_splits[0], train_val_test_splits[1], train_val_test_splits[2], batch_size)

    if len(val) == 0:
        val=None

    return train, val, test, batch_size




@gin.configurable
def train(encoder, decoder, n_epochs, train_loader, batch_size, mode, masked_percentage=15 ,model = None, log_dir=None, val_loader=None, args=None):

    writer = SummaryWriter(log_dir)

    if args is not None:
        shutil.copyfile(args.gin ,os.path.join(writer.log_dir, "gin.config"))

    if model is None:
        model = Encoder_Decoder_Model(encoder=encoder, decoder=decoder,batch_size=batch_size)

    match mode:
        case "masked":
            training = MaskedTraining(model, n_epochs, masked_percentage, writer)
            #training = training(model, n_epochs, masked_percentage, writer)
        case "autoencode":
            training = AutoEncoder(model, n_epochs, writer)
            #training = training(model, n_epochs, writer)
        case "TE":
            training = TETraining(model, n_epochs, writer)
            #training = training(model, n_epochs, writer)

    


    training.run(train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(writer.log_dir, "model"))

    return model, log_dir, training.device, mode


def test(test_loader, model, log_dir, device, test_mode, test_name):
    if not os.path.exists(os.path.join(log_dir, "test")):
        os.makedirs(os.path.join(log_dir, "test"))

    model.to(device)

    for i, batch in enumerate(test_loader):
        batch.to(device)
        if test_mode == "TE":
            pred_MRL = model(batch)
            df = pd.DataFrame({'pred_mrl':pred_MRL.detach().cpu().numpy(), 'true_mrl':batch.mrl.detach().cpu().numpy(), 'id':batch.ID.cpu()})
            df.to_csv(os.path.join(log_dir, "test", "pred_"+test_name+".csv"), mode='a', header=(i==0), index=False)

        if test_mode == "SeqStruc":
            break
        # TODO



        
    




    