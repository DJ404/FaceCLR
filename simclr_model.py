import torch
import lightning as L
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from simclr_loss import SimCLR_Loss


class SimCLR_Model(L.LightningModule):
    """The encoder model for the SimCLR procedure as a pytorch lightning module.
    
    The backbone structure of the neural net is the ConvNeXt base model. AdamW optimizer and cosine annealing learning rate scheduler is used for training.
    The loss is the NT-Xent loss from the paper.

    Attributes:
        args (str): args from the parser to pass to the constructor.
        classes (int): dimension of the output layer. This number will define the size of the embedding vector.
        encoder: the backbone neural net model used for encoding.
        loss: the NT-Xent loss implemented as a custom class.
        
    """
    
    def __init__(self, args, n_classes):
        """Creates an instance of the SimCLR model where the size of the output layer is defiend by n_classes.

        Args:
            args (str): pass the "batch_size", "nodes" and "gpus" as args to a instance. It is needed for the loss class.
            n_classes (int): number of nodes in the output layer. Is equal to the dimension of the embeddign vector.
        """
        
        super().__init__()
        
        self.args = args
        self.classes = n_classes
        self.encoder = convnext_base(num_classes=self.classes)
        self.loss = SimCLR_Loss(self.args.batch_size, self.args.nodes, self.args.gpus, self.args.temperature)
        
        
    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (tensor): Input to the model.

        Returns:
            (tensor): Output of the model. The output is the embedding of the input image.
        """
        
        return self.encoder(x)
        
        
    def configure_optimizers(self):
        """The optimizer and learning rate scheduler used for training are defined here.

        The AdamW optimizer and cosine annealing learning rate schedule is used.
        
        Returns:
            dict: The dictionary includes the chosen optimizer and scheduler and their corresponding keys.
        """
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        
    def _step(self, batch, log_loss="train"):
        """perfoms one training or validation step including the loss calculation.

        Different augmented versions of the images are encoded seperately. The loss gets calculated using all embeddings across all nodes.
        
        Args:
            batch: one batch from the dataset.
            log_loss (str, optional): specifies the logged loss, can be either "train" or "val". Defaults to "train".

        Returns:
            (tensor): loss for the batch.
        """
        x, y = batch
        emb_i = self.encoder(x)
        emb_j = self.encoder(y)
        
        all_emb_i= self.all_gather(emb_i, sync_grads=True).view(-1, self.classes).contiguous()
        all_emb_j= self.all_gather(emb_j, sync_grads=True).view(-1, self.classes).contiguous()
        
        loss = self.loss(all_emb_i, all_emb_j)
        self.log(log_loss+"_loss", loss, on_epoch=True, sync_dist=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        """Wrapper function to call a the step function used for training.
        
        The logged loss will be the train loss

        Args:
            batch: one batch from the dataset.
            batch_idx: the corresponding batch index.

        Returns:
            (tensor): the training loss for one batch using the step function.
        """
        
        return self._step(batch, log_loss="train")
    
    
    def validation_step(self, batch, batch_idx):
        """Wrapper function to call a the step function used for validation.
        
        The logged loss will be the validation loss

        Args:
            batch: one batch from the dataset.
            batch_idx: the corresponding batch index.

        Returns:
            (tensor): the validation loss for one batch using the step function.
        """
        return self._step(batch, log_loss="val")