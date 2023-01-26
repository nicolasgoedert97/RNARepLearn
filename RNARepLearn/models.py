from RNARepLearn.modules import LinearEmbedding, RPINetEncoder, AttentionDecoder
import torch

def TestModel():
    layers = []
    layers.append(LinearEmbedding(4, 32))
    layers.append(RPINetEncoder(32, 32, 3))
    layers.append(AttentionDecoder(32, 4))
    model = torch.nn.Sequential(*layers)
    return model




