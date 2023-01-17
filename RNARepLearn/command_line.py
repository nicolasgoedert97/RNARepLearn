import RNARepLearn
import argparse
import gin
from torch.utils.tensorboard import SummaryWriter
from RNARepLearn.datasets import CombinedRfamDataset
from RNARepLearn.utils import train_val_test_loaders
from RNARepLearn.models import TestModel
from RNARepLearn.train import MaskedTraining

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin')

    args = parser.parse_args()

    gin.parse_config_file(args.gin)

    train_loader, val_loader, test_loader = setup_data()

    train(train_loader=train_loader, val_loader=val_loader)




@gin.configurable
def setup_data(rfam_basepath, rfams, train_val_test_splits, batch_size, seq_length_lim=None, dataset_name=None):
    dataset = None

    print("Dataset name: "+str(dataset_name))
    print(" ".join([rfam_basepath, rfams[0], str(dataset_name), str(seq_length_lim)]))
    dataset = CombinedRfamDataset(rfam_basepath, rfams, dataset_name, seq_length_lim)

    print("Dataset length: "+str(len(dataset)))

    train, val, test = train_val_test_loaders(dataset, train_val_test_splits[0], train_val_test_splits[1], train_val_test_splits[2], batch_size)

    if len(val) == 0:
        val=None

    return train, val, test




@gin.configurable
def train(training, model, n_epochs, masked_percentage, train_loader, log_dir=None, val_loader=None):

    # Set training and model TODO Find a way to cleanly do this in config.gin (imports dont work)
    match training:
        case "masked":
            training = MaskedTraining

    match model:
        case "test":
            model = TestModel()

    writer = SummaryWriter(log_dir)

    training = training(model, n_epochs, masked_percentage, writer)

    training.run(train_loader, val_loader)




    