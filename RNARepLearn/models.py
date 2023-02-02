from RNARepLearn.modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder, CNN_Encoder, GCN_Encoder
import torch
import gin

@gin.configurable
class Encoder_Decoder_Model(torch.nn.Module):
    def __init__(self,input_channels, output_channels ,encoder, decoder, encoding_layers, encoding_kernel_size , batch_size):
        super().__init__()
        layers = []

        if encoder == 'RPI':
            layers.append(RPINetEncoder(input_channels,output_channels, encoding_layers, encoding_kernel_size))
            output_channels = output_channels*2

        if encoder == "CNN":
            layers.append(CNN_Encoder(input_channels,output_channels, encoding_kernel_size))

        if encoder == "GCN":
            layers.append(GCN_Encoder(input_channels, output_channels, encoding_kernel_size, encoding_layers))

        
        if decoder == "TE":
            layers.append(TE_Decoder(input_channels=output_channels, batch_size=batch_size))

        if decoder == "SeqStruc":
            layers.append(AttentionDecoder(output_channels, input_channels))

        self.model = torch.nn.Sequential(*layers).double()
    
    def forward(self, batch):

        return self.model(batch)
        







