import argparse

import torch, torchvision, torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model import Baseline_Model
from data_augmentation import DataAug
from lfw_n_classes import LFWPeopleNClasses, TransformationDataset



def main():
    """ main script to run the training and evaluation for the baseline model.
    
    Handles the command line inputs and also the dataset is defined here.
    """
    
    parser = argparse.ArgumentParser(description="ConvNext Baseline Model")
    
    parser.add_argument("-b","--batch-size",  type=int, required=True, help="Number of samples in each mini-batch")
    parser.add_argument("-lr", "--learning-rate", default=1e-5, type=float, help="Specify starting learning rate")
    parser.add_argument("-wd", "--weight-decay", default=1e-2, type=float, help="Specify weight decay")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="Number of classes for final layer")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of total epochs")
    args = parser.parse_args()
    
    #to be reproduceable
    generator = torch.Generator().manual_seed(42)
    
    dataset = LFWPeopleNClasses("./data", 10, "10fold")
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator)
    train_set_transformed = TransformationDataset(train_set,DataAug(224))
    val_set_transformed = TransformationDataset(val_set, transforms.ToTensor())
    test_set_transformed = TransformationDataset(test_set, transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set_transformed, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set_transformed, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set_transformed, batch_size=args.batch_size, shuffle=False)
    
    
    args.num_classes = dataset.classes

    
    model = Baseline_Model(args)
    trainer = L.Trainer(max_epochs=args.epochs) #callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()