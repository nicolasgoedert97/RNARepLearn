import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch
import gin
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import mask_batch, reconstruct_bpp, EarlyStopper, add_backbone
from .losses import BppReconstructionLoss




class Training:
    def __init__(self, model, n_epochs, writer=None, lr=0.01, weight_decay=5e-4, parallel=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = model.double()
        self.model.to(self.device)
        print("Learning rate: "+str(lr))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.n_epochs = n_epochs
        self.writer = writer
        self.parallel = parallel

        if self.writer is None:
            print("Creating new SummaryWriter")
            self.writer = SummaryWriter()
        
        print("Model: \n"+str(model))
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


@gin.configurable
class MaskedTraining(Training):
    def __init__(self, model, n_epochs, masked_percentage, writer, lr=0.002, weight_decay=5e-4 , mask=True, mask_seq=True, mask_struc=True, validation_steps=5, node_loss_weight=1, edge_loss_weight=1, early_stopping_patience=10, add_bb=False, print_out=False, parallel=False):
        super().__init__(model, n_epochs, writer, lr, weight_decay, parallel)
        self.masked_percentage = masked_percentage
        self.mask = mask
        self.mask_seq = mask_seq
        self.mask_struc = mask_struc
        self.val_steps = validation_steps
        self.edge_loss_weight = edge_loss_weight
        self.node_loss_weight = node_loss_weight
        self.add_bb = add_bb
        self.patience = early_stopping_patience

    
    def run(self, data_loader, val_loader=None):
        print("Training running on device: "+str(self.device))
        
        cel_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        early_stop = EarlyStopper(patience=self.patience)

        #self.model.train()

        step=0
        last_loss = 100
        batches_saved = 10
        for epoch in range(self.n_epochs):
            epoch_loss = []
            for i,batch in enumerate(data_loader):
                step+=1

                if self.add_bb:
                    batch = add_backbone(batch, True)

                if self.parallel:
                    true_x = torch.clone(torch.cat([data.x[:,:4] for data in batch]))
                    true_weights = torch.clone(torch.cat([data.edge_weight for data in batch]))
                    train_mask = torch.cat(torch.tensor([mask_batch(graph, self.masked_percentage, self.mask, self.mask_seq, self.mask_struc) for graph in batch]))
                else:
                    true_x = torch.clone(batch.x[:,:4])#[:,:4]
                    true_weights = torch.clone(batch.edge_weight)

                    train_mask = mask_batch(batch,self.masked_percentage, self.mask, self.mask_seq, self.mask_struc)

                
                if not self.parallel:
                    batch.to(self.device)

                self.optimizer.zero_grad()

                nucs, bpp = self.model(batch)

                node_loss = cel_loss(nucs.cpu()[train_mask],true_x[train_mask])
                edge_loss = kl_loss(bpp[train_mask].cpu(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0])))[train_mask] )

                
                loss = node_loss * self.node_loss_weight + edge_loss * self.edge_loss_weight

                last_loss = loss.item()

                loss.backward()
                self.optimizer.step()
                
                node_accuracy = int((nucs.cpu()[train_mask].argmax(dim=1)==true_x[train_mask].argmax(dim=1)).sum()) / sum(train_mask)
                
                if step % 10 == 0:

                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e Nucleotide-Loss: % 2.2e Edge-Loss: % 2.2e  Memory % 4d' % (epoch + 1, self.n_epochs, 
                                                                        i + 1, len(data_loader), 
                                                                        loss.item(),node_loss.item(),edge_loss.item(), torch.cuda.memory_allocated()))
                    self.writer.add_scalar("Loss/train", loss.item(), step)
                    self.writer.add_scalar("Loss_nodes/train", node_loss.item(), step)
                    self.writer.add_scalar("Loss_edges/train", edge_loss.item(), step)
                    self.writer.add_scalar("Node_Accuracy/train", node_accuracy, step)
                    epoch_loss.append(loss.item())

            epoch_loss = (sum(epoch_loss)/len(epoch_loss) if len(epoch_loss)>0 else 0)
            self.writer.add_scalar("Epoch_Loss/train", epoch_loss , epoch)

            if val_loader is not None:
                self.model.eval()
                node_loss_val_list = []
                edge_loss_val_list = []
                node_accuracy_val_list = []

                

                with torch.no_grad():
                    for i,batch in enumerate(val_loader):
                        true_x = torch.clone(batch.x[:,:4])#[:,:4]
                        true_weights = torch.clone(batch.edge_weight)

                        train_mask = mask_batch(batch,self.masked_percentage, self.mask, self.mask_seq, self.mask_struc)
                        batch.to(self.device)
                
                        nucs, bpp = self.model(batch)
                        edge_loss_val_step = kl_loss(bpp[train_mask].cpu(), torch.tensor(reconstruct_bpp(batch.edge_index.detach().cpu(), true_weights.cpu(), (batch.x.shape[0],batch.x.shape[0])))[train_mask] )
                        node_loss_val_step = cel_loss(nucs.cpu()[train_mask],true_x[train_mask])
                        node_accuracy_val_step = int((nucs.cpu()[train_mask].argmax(dim=1)==true_x[train_mask].argmax(dim=1)).sum()) / sum(train_mask)
                        
                        node_loss_val_list.append(node_loss_val_step.item())
                        edge_loss_val_list.append(edge_loss_val_step.item())
                        node_accuracy_val_list.append(node_accuracy_val_step.item())

                    node_loss_val = sum(node_loss_val_list)/len(node_loss_val_list) if len(node_loss_val_list)>0 else 0
                    edge_loss_val = sum(edge_loss_val_list)/len(edge_loss_val_list) if len(edge_loss_val_list)>0 else 0
                    node_accuracy_val = sum(node_accuracy_val_list)/len(node_accuracy_val_list) if len(node_accuracy_val_list)>0 else 0

                    val_loss = node_loss_val * self.node_loss_weight + edge_loss_val * self.edge_loss_weight

                    self.writer.add_scalar("Epoch_Loss/val", val_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_loss, step)
                    self.writer.add_scalar("Loss_edges/val", edge_loss_val, step)
                    self.writer.add_scalar("Node_Accuracy/val", node_accuracy_val, step)
                    self.writer.add_scalar("Loss_nodes/val", node_loss_val, step)

                    print('\r[Epoch %4d/%4d] Epoch-Loss: % 2.2e Epoch-Loss Val: % 2.2e Nucleotide-Loss Val: % 2.2e Edge-Loss Val: % 2.2e  Memory % 4d' % (epoch + 1, self.n_epochs,  
                                                                        epoch_loss,val_loss, node_loss_val, edge_loss_val, torch.cuda.memory_allocated()))

                    if early_stop.early_stop(val_loss):
                        print("Earlystopped after Epoch "+str(epoch)+" with patience of "+str(early_stop.patience))
                        break

                self.model.train()

            self.writer.flush()
            torch.save(self.model.state_dict(), os.path.join(self.writer.log_dir, "model_epoch"+str(epoch)))
            
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

    def __init__(self, model, n_epochs, writer, lr=0.00002, weight_decay=5e-4 ,parallel=False, early_stopping_patience=None):
        super().__init__(model, n_epochs, writer, lr, weight_decay, parallel)
        self.early_stopping_patience = early_stopping_patience

    def run(self, train_loader, val_loader=None):
        print("Training running on device: "+str(self.device))
        mse_loss = torch.nn.MSELoss()

        self.model.train()

        if self.early_stopping_patience is not None:
            early_stop = EarlyStopper(patience=self.early_stopping_patience)

        step=0
        for epoch in range(self.n_epochs):
            epoch_loss = []
            for idx, batch in enumerate(train_loader):

                batch.to(self.device)
                self.optimizer.zero_grad()


                pred_MRL = self.model(batch)

                loss = mse_loss(torch.unsqueeze(pred_MRL, 0), batch.mrl.double())
                epoch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(train_loader), 
                                                                        loss.item()))
                    self.writer.add_scalar("Loss/train", loss.item(), step)
                step+=1

            if val_loader is not None:
                self.model.eval()
                val_loss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.to(self.device)

                        pred_MRL = self.model(batch)

                        val_loss_step = mse_loss(torch.unsqueeze(pred_MRL, 0), batch.mrl.double())
                        val_loss.append(val_loss_step.item())


                self.model.train()
                val_loss = sum(val_loss)/len(val_loss)
                loss = sum(epoch_loss)/len(epoch_loss)

                self.writer.add_scalar("Epoch_Loss/val", val_loss, epoch)
                self.writer.add_scalar("Epoch_Loss/train", loss, epoch)

                if self.early_stopping_patience is not None:
                    if early_stop.early_stop(val_loss):
                        print("Earlystopped after Epoch "+str(epoch)+" with patience of "+str(early_stop.patience))
                        break
                
            self.writer.flush()
            torch.save(self.model.state_dict(), os.path.join(self.writer.log_dir, "model_epoch"+str(epoch)))
        self.writer.close()


class AffinityTraining(Training):

    def __init__(self, model, n_epochs, writer, lr=0.00002, weight_decay=5e-4 ,parallel=False, early_stopping_patience=None, bce_loss_weight=None):
        super().__init__(model, n_epochs, writer, lr, weight_decay, parallel)
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = bce_loss_weight

    def run(self, train_loader,  n_prots, val_loader=None):
        print("Training running on device: "+str(self.device))
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.class_weight)

        self.model.train()

        if self.early_stopping_patience is not None:
            early_stop = EarlyStopper(patience=self.early_stopping_patience)

        step=0
        for epoch in range(self.n_epochs):
            epoch_loss = []
            for idx, batch in enumerate(train_loader):

                batch.to(self.device)
                self.optimizer.zero_grad()


                pred_affinity = self.model(batch)

                loss = bce_loss(pred_affinity.cpu(), batch.binding_prots.view(len(batch.ID), n_prots).double().cpu())
                epoch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if idx % 10 == 0:
                    print('\r[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e' % (epoch + 1, self.n_epochs, 
                                                                        idx + 1, len(train_loader), 
                                                                        loss.item()))
                    self.writer.add_scalar("Loss/train", loss.item(), step)
                step+=1

            if val_loader is not None:
                self.model.eval()
                val_loss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.to(self.device)

                        pred_affinity = self.model(batch)

                        val_loss_step = bce_loss(pred_affinity.cpu(), batch.binding_prots.view(len(batch.ID), n_prots).double().cpu())
                        val_loss.append(val_loss_step.item())


                self.model.train()
                val_loss = sum(val_loss)/len(val_loss)
                loss = sum(epoch_loss)/len(epoch_loss)

                self.writer.add_scalar("Epoch_Loss/val", val_loss, epoch)
                self.writer.add_scalar("Epoch_Loss/train", loss, epoch)

                if self.early_stopping_patience is not None:
                    if early_stop.early_stop(val_loss):
                        print("Earlystopped after Epoch "+str(epoch)+" with patience of "+str(early_stop.patience))
                        break
                
            self.writer.flush()
            torch.save(self.model.state_dict(), os.path.join(self.writer.log_dir, "model_epoch"+str(epoch)))
        self.writer.close()
