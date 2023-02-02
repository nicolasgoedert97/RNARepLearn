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
from RNARepLearn.utils import train_val_test_loaders
from RNARepLearn.models import Encoder_Decoder_Model
from RNARepLearn.train import MaskedTraining, AutoEncoder, TETraining

mode = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin')

    args = parser.parse_args()

    gin.parse_config_file(args.gin)
    set_mode()

    train_loader, val_loader, test_loader, batch_size = setup_data()

    model, log_dir, train_device = train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, args=args)

    test(test_loader, model, log_dir, train_device)

@gin.configurable
def set_mode(app_mode):
    global mode
    mode = app_mode

@gin.configurable
def setup_data(basepath, train_val_test_splits, batch_size, rfams=None, seq_length_lim=None, dataset_name=None):
    dataset = None

    print("Dataset name: "+str(dataset_name))

    match mode:
        case "RFAM":
            dataset = CombinedRfamDataset(basepath, rfams, dataset_name, seq_length_lim)
        case "TE":
            dataset = Dataset_UTR5(basepath)
        case None:
            raise Exception("No application mode defined")
        
    print("Dataset length: "+str(len(dataset)))

    train, val, test = train_val_test_loaders(dataset, train_val_test_splits[0], train_val_test_splits[1], train_val_test_splits[2], batch_size)

    if len(val) == 0:
        val=None

    return train, val, test, batch_size


@gin.configurable
def train(encoder, decoder, n_epochs, train_loader, batch_size, mode, masked_percentage=15 ,log_dir=None, val_loader=None, args=None):

    writer = SummaryWriter(log_dir)

    shutil.copyfile(args.gin ,os.path.join(writer.log_dir, "gin.config"))

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

    return model, log_dir, training.device


def test(test_loader, model, log_dir, device):
    if not os.path.exists(os.path.join(log_dir, "test")):
        os.makedirs(os.path.join(log_dir, "test"))

    model.to(device)

    for i, batch in enumerate(test_loader):
        batch.to(device)
        pred_MRL = model(batch)
        df = pd.DataFrame({'pred_mrl':pred_MRL.detach().cpu().numpy(), 'true_mrl':batch.mrl.detach().cpu().numpy(), 'id':batch.ID})
        df.to_csv(os.path.join(log_dir, "test", "pred.csv"), mode='a', header=(i==0), index=False)
    



    