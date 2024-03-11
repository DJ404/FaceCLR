import torch.nn as nn, torch
import lightning as L
from torchvision.models import convnext_small
from torchmetrics import Accuracy, JaccardIndex

class Baseline_Model(L.LightningModule):
    """ The baseline neural network model as a pytorch lightning module. 
    
    The structure of the neural net and its training procedure is defined here.

    Attributes:
        args (str): args from the parser to pass to the constructor.
        model: the neural net model architecture used for training.
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
    
    def __init__(self, args):
        """Creates an instance of the basemodel class with the number of nodes in the output layer passed by args.

        Args:
            args (str): pass the args.num_classes to the baseline model. It represents the number of classes or output neurons.
        """
        super().__init__()
        
        self.args = args
        self.model = convnext_small(num_classes=self.args.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        self.train_topk1 = Accuracy(task="multiclass", num_classes=self.args.num_classes)
        self.train_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=self.args.num_classes)
        self.train_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        self.val_topk1 = Accuracy(task="multiclass", num_classes=self.args.num_classes)
        self.val_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=self.args.num_classes)
        self.val_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        self.test_topk1 = Accuracy(task="multiclass", num_classes=self.args.num_classes)
        self.test_topk5 = Accuracy(task="multiclass", top_k=5, num_classes=self.args.num_classes)
        self.test_index = JaccardIndex(task="multiclass", num_classes=args.num_classes, average="micro")
        
        
    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (tensor): Input to the model.

        Returns:
            (tensor): Output of the model.
        """
        
        return self.model(x)
        
        
    def configure_optimizers(self):
        """The optimizer and learning rate scheduler used for training are defined here.

        Returns:
            dict: The dictionary includes the chosen optimizer and scheduler and their corresponding keys.
        """
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        
    def training_step(self, batch, batch_idx):
        """Defines one training step of the model.
        
        It includes the loss calculation, logging and how a forward pass is conducted.

        Args:
            batch: One batch from the dataset.
            batch_idx: The corresponding index for the batch.
        Returns:
            tensor: the training loss for the batch
        """
        
        images, targets = batch
        pred = self.model(images)
        loss = self.loss(pred, targets)
        self.train_topk1(pred, targets)
        self.train_topk5(pred, targets)
        self.train_index(pred, targets)
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
        
        
        images, targets = batch
        pred = self.model(images)
        loss = self.loss(pred, targets)
        self.val_topk1(pred, targets)
        self.val_topk5(pred, targets)
        self.val_index(pred, targets)
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
        images, targets = batch
        pred = self.model(images)
        loss = self.loss(pred, targets)
        self.test_topk1(pred, targets)
        self.test_topk5(pred, targets)
        self.test_index(pred, targets)
        self.log_dict({"test_loss": loss, "test_acc_top1": self.test_topk1, "test_acc_top5": self.test_topk5, "test_jaccard_index": self.test_index}, on_epoch=True, on_step=False)
        return loss
