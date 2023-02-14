import torch
from torch_geometric.utils import to_dense_batch
import gin
from torch.utils.tensorboard import SummaryWriter
from .utils import mask_batch, reconstruct_bpp
from .losses import BppReconstructionLoss




class Training:
    def __init__(self, model, n_epochs, writer=None, lr=0.01, weight_decay=5e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.model = model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.n_epochs = n_epochs
        self.writer = writer

        if self.writer is None:
            print("Creating new SummaryWriter")
            self.writer = SummaryWriter()


@gin.configurable
class MaskedTraining(Training):
    def __init__(self, model, n_epochs, masked_percentage, writer, lr=0.01, weight_decay=5e-4):
        super().__init__(model, n_epochs, writer, lr, weight_decay)
        self.masked_percentage = masked_percentage
    
    def run(self, data_loader, val_loader=None):
        print("Training running on device: "+str(self.device))
        cel_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.model.train()

        step=0
        for epoch in range(self.n_epochs):
            for idx, batch in enumerate(data_loader):

                true_x = torch.clone(batch.x)
                true_weights = torch.clone(batch.edge_weight)

                train_mask = mask_batch(batch,self.masked_percentage, False)

                batch.to(self.device)
                self.optimizer.zero_grad()

                nucs, bpp = self.model(batch)

                node_loss = cel_loss(nucs.cpu()[train_mask],true_x[train_mask])
                edge_loss = kl_loss(bpp[train_mask][:,train_mask].cpu().log(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0])))[train_mask][:,train_mask] )
                
                loss = node_loss + edge_loss

                loss.backward()
                self.optimizer.step()
                
                node_accuracy = int((nucs.cpu()[train_mask].argmax(dim=1)==true_x[train_mask].argmax(dim=1)).sum()) / sum(train_mask)

                node_accuracy_val = None
                edge_loss_val = None

                if val_loader is not None:
                    self.model.eval()
                    node_accuracy_val = []
                    edge_loss_val = []

                    for batch in val_loader:
                        true_x = torch.clone(batch.x)
                        true_weights = torch.clone(batch.edge_weight)

                        train_mask = mask_batch(batch,self.masked_percentage, False)
                        batch.to(self.device)
                        self.optimizer.zero_grad()
                
                        nucs, bpp = self.model(batch)
                        edge_loss_val_step = kl_loss(bpp[train_mask][:,train_mask].cpu().log(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0])))[train_mask][:,train_mask] )
                        node_accuracy_val.append(int((nucs.cpu()[train_mask].argmax(dim=1)==true_x[train_mask].argmax(dim=1)).sum()) / sum(train_mask))
                        edge_loss_val.append(float(edge_loss_val_step))

                    node_accuracy_val = sum(node_accuracy_val)/len(node_accuracy_val)
                    edge_loss_val = sum(edge_loss_val)/len(edge_loss_val)

                    self.model.train()
                    
                
                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e Nucleotide-Loss: % 2.2e Edge-Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(data_loader), 
                                                                        loss.item(),node_loss.item(),edge_loss.item()))
                self.writer.add_scalar("Loss/train", loss.item(), step)
                self.writer.add_scalar("Loss_nodes/train", node_loss.item(), step)
                self.writer.add_scalar("Loss_edges/train", edge_loss.item(), step)
                self.writer.add_scalar("Loss_edges/val", edge_loss_val, step)
                self.writer.add_scalar("Node_Accuracy/train", node_accuracy, step)
                self.writer.add_scalar("Node_Accuracy/val", node_accuracy_val, step)
                step+=1
            self.writer.flush()
        self.writer.close()


class AutoEncoder(Training):
    
    def run(self, data_loader, val_loader=None):
        cel_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.model.train()

        step=0
        for epoch in range(self.n_epochs):
            for idx, batch in enumerate(data_loader):

                true_x = torch.clone(batch.x)
                true_edges = torch.clone(batch.edge_weight)

                batch.to(self.device)
                self.optimizer.zero_grad()
                
                nucs, bpp = self.model(batch)

                node_loss = cel_loss(nucs.cpu(),true_x)
                edge_loss = kl_loss(to_dense_batch(bpp.cpu(), batch.batch.cpu())[0].log(), to_dense_batch(torch.tensor(reconstruct_bpp(batch.edge_index.cpu(), true_edges, (len(bpp),len(bpp)))), batch.batch.cpu())[0])
                
                loss = node_loss*100 + edge_loss

                loss.backward()
                self.optimizer.step()
                
                node_accuracy = int((nucs.cpu().argmax(dim=1)==true_x.argmax(dim=1)).sum()) / nucs.shape[0]

                node_accuracy_val = None
                edge_accuracy_val = None
                if val_loader is not None:
                    self.model.eval()
                    node_accuracy_val = []
                    edge_accuracy_val = []
                    for batch in val_loader:
                        true_x = torch.clone(batch.x)
                        
                        batch.to(self.device)
                        self.optimizer.zero_grad()
                
                        nucs, bpp = self.model(batch)
                        node_accuracy_val.append(int((nucs.cpu().argmax(dim=1)==true_x.argmax(dim=1)).sum()) / nucs.shape[0])
                        edge_accuracy_val.append(int(kl_loss(to_dense_batch(bpp.cpu(), batch.batch.cpu())[0].log(), to_dense_batch(torch.tensor(reconstruct_bpp(batch.edge_index.cpu(), batch.edge_weight.cpu(), (len(bpp),len(bpp)))), batch.batch.cpu())[0])))

                    node_accuracy_val = sum(node_accuracy_val)/len(node_accuracy_val)
                    edge_accuracy_val = sum(edge_accuracy_val)/len(edge_accuracy_val)
                    
                    
                    self.model.train()
                
                
                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e Nucleotide-Loss: % 2.2e Edge-Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(data_loader), 
                                                                        loss.item(),node_loss.item(),edge_loss.item()))
                self.writer.add_scalar("Loss/train", loss.item(), step)
                self.writer.add_scalar("Loss_nodes/train", node_loss.item(), step)
                self.writer.add_scalar("Loss_edges/train", edge_loss.item(), step)
                self.writer.add_scalar("Loss_edges/val", edge_accuracy_val, step)
                self.writer.add_scalar("Node_Accuracy/train", node_accuracy, step)
                self.writer.add_scalar("Node_Accuracy/val", node_accuracy_val, step)
                step+=1
            self.writer.flush()
        self.writer.close()


class TETraining(Training):

    def run(self, train_loader, val_loader=None):
        print("Training running on device: "+str(self.device))
        mse_loss = torch.nn.MSELoss()

        self.model.train()

        step=0
        for epoch in range(self.n_epochs):
            for idx, batch in enumerate(train_loader):

                batch.to(self.device)
                self.optimizer.zero_grad()


                pred_MRL = self.model(batch)

                loss = mse_loss(torch.unsqueeze(pred_MRL, 0), batch.mrl.double())

                loss.backward()
                self.optimizer.step()

                val_loss = None
                if val_loader is not None:
                    self.model.eval()

                    batch = next(iter(val_loader))
                    batch.to(self.device)

                    pred_MRL = self.model(batch)

                    val_loss = mse_loss(torch.unsqueeze(pred_MRL, 0), batch.mrl.double())

                    self.model.train()


                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e Validation-Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(train_loader), 
                                                                        loss.item(),val_loss.item()))
                self.writer.add_scalar("Loss/train", loss.item(), step)
                self.writer.add_scalar("Loss/val", val_loss.item(), step)

                step+=1
            self.writer.flush()
        self.writer.close()
