import torch, torch.nn as nn
import lightning as L
from simclr_model import SimCLR_Model
from torchmetrics import Accuracy, JaccardIndex


class LinearModel(L.LightningModule):
    """ The model used for linear evaluation as a pytorch lightning module. 
    
    The structure of the neural net and its training procedure is defined here.

    Attributes:
        args (str): args from the parser to pass to the constructor.
        simclr_model: the embedding model used to perform linear evaluation on.
        model: single linear layer for linear evaluation.
        loss: the loss used for training is a torch module.
        train_topk1: torchmetrics accuarcy class measures the training top-1 error.
        train_topk5: torchmetrics accuarcy class measures the training top-5 error.
        train_index: torchmetrics JaccardIndex class measures the training JI.
        val_topk1: torchmetrics accuarcy class measures the validation top-1 error.
        val_topk5: torchmetrics accuarcy class measures the validation top-5 error.
        val_index: torchmetrics JaccardIndex class measures the validation JI.
        test_topk1: torchmetrics accuarcy class measures the test top-1 error.
        test_topk5: torchmetrics accuarcy class measures the test top-5 error.
        test_index: torchmetrics JaccardIndex class measures the test JI.
        
    """
    def __init__(self, args, emb_model):
        """creates an instance of the linear evalation model with the given encoder model.

        Args:
            args (str): pass the num_classes variable being the number of classes used for classification.
            emb_model (LightningModule): pretrained encoder model used for creating the embeddings.
        """
        super().__init__()
        
        self.args = args
        self.simclr_model = emb_model
        self.simclr_model.eval()
        self.model = nn.Linear(self.args.embedding_space, self.args.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        self.train_topk1 = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.train_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=args.num_classes)
        self.train_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        self.val_topk1 = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.val_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=args.num_classes)
        self.val_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        self.test_topk1 = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.test_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=args.num_classes)
        self.test_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        
    def forward(self, x):
        """defines the forward pass of the model.
        
        The input gets passed to the encoder model with frozen weights to create the embedding before passing the input to the linear model.

        Args:
            x (tensor): Input to the encoder and linear model.
        Returns:
            (tensor): Output of the linear model.
        """
        
        with torch.no_grad():
            embedding = self.simclr_model(x)
        return self.model(embedding)
        
        
    def configure_optimizers(self):
        """The optimizer and learning rate scheduler used for training are defined here.

        The AdamW optimizer and cosine annealing learning rate schedule is used.
        
        Returns:
            dict: The dictionary includes the chosen optimizer and scheduler and their corresponding keys.
        """
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
        
    def training_step(self, batch, batch_idx):
        """Defines one training step of the model.
        
        It includes the loss calculation, logging and calls the forward method.

        Args:
            batch: One batch from the dataset.
            batch_idx: The corresponding index for the batch.
        Returns:
            tensor: the training loss for the batch
        """
        
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        self.train_topk1(out, y)
        self.train_topk5(out, y)
        self.train_index(out, y)
        self.log_dict({"train_loss": loss, "train_acc_top1": self.train_topk1, "train_acc_top5": self.train_topk5, "train_jaccard_index": self.train_index}, on_epoch=True, on_step=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Defines one validation step of the model.
        
        Analogous to the training step function.

        Args:
            batch: One batch from the dataset.
            batch_idx: The corresponding index for the batch.
        Returns:
            tensor: the validation loss for the batch
        """
        
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        self.val_topk1(out, y)
        self.val_topk5(out, y)
        self.val_index(out, y)
        self.log_dict({"val_loss": loss, "val_acc_top1": self.val_topk1, "val_acc_top5": self.val_topk5, "val_jaccard_index": self.val_index}, on_epoch=True, on_step=False)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        """Defines one test step of the model.
        
        Analogous to the training step function.

        Args:
            batch: One batch from the dataset.
            batch_idx: The corresponding index for the batch.
        Returns:
            tensor: the test loss for the batch
        """
        
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        self.test_topk1(out, y)
        self.test_topk5(out, y)
        self.test_index(out, y)
        self.log_dict({"test_loss": loss, "test_acc_top1": self.test_topk1, "test_acc_top5": self.test_topk5, "test_jaccard_index": self.test_index}, on_epoch=True, on_step=False)
        return loss
