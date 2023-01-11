import torch
from .utils import mask_batch, reconstruct_bpp

class MaskedTraining():

    def __init__(self, model, n_epochs, masked_percentage, writer, lr=0.01, weight_decay=5e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.model = model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.n_epochs = n_epochs
        self.masked_percentage = masked_percentage
        self.writer = writer
    
    def run(self, data_loader):
        cel_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.model.train()

        for epoch in range(self.n_epochs):
            for idx, batch in enumerate(data_loader):
                
                true_x = torch.clone(batch.x)
                true_edges = torch.clone(batch.edge_weight)

                nuc_mask, edge_mask = mask_batch(batch,self.masked_percentage)
                batch.to(self.device)

                self.optimizer.zero_grad()
                
                nucs, bpp = self.model(batch)

                node_loss = cel_loss(nucs.cpu()[nuc_mask],true_x[nuc_mask])
                edge_loss = kl_loss(bpp.cpu()[nuc_mask].log() , torch.tensor(reconstruct_bpp(batch.edge_index.cpu(), true_edges, (len(bpp),len(bpp)))[nuc_mask]))
                
                loss = node_loss + edge_loss
                
                self.writer.add_scalar("Loss/train", loss, epoch)
                self.writer.add_scalar("Loss_nodes/train", node_loss, epoch)
                self.writer.add_scalar("Loss_edges/train", edge_loss, epoch)

                loss.backward()
                self.optimizer.step()
                
                
                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e Nucleotide-Loss: % 2.2e Edge-Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(data_loader), 
                                                                        loss.item(),node_loss.item(),edge_loss.item()))
        writer.flush()