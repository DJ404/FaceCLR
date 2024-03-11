import argparse
import os

import torch, torch.nn as nn
import lightning as L
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from simclr_model import SimCLR_Model


from celebADataset import CelebADataset
from augmentation import Augmentation



def main():
    """Main script for runnign the SimCLR encoder training.
    
    Handles the command line inputs and also the dataset is defined here.
    The ConvNeXt backbone is used in combination with the CelebA dataset.
    """
    
    
    parser = argparse.ArgumentParser(description="Face Embeddings with Contrastive Learning")
    
    parser.add_argument("-b","--batch-size",  type=int, required=True, help="Number of samples in each mini-batch")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of total epochs")
    parser.add_argument("-lr", "--learning-rate", default=5e-4, type=float, help="Specify starting learning rate")
    parser.add_argument("-wd", "--weight-decay", default=1e-4, type=float, help="Specify weight decay")
    parser.add_argument("--embedding_space", type=int, default=512, help="The size of the embedding vector")
    parser.add_argument("-p", "--datadir", default="../ContrastiveFaceEmb128/data", help="Path to the datasets")
    parser.add_argument("-t", "--temperature", default=0.5, type=int, help="Temperature scale for the loss")
    parser.add_argument("-n", "--nodes", default=1, type=int, help="Number of nodes for DDP")
    
    args = parser.parse_args()


    args.gpus = max(torch.cuda.device_count(), 1)


    dataset = CelebADataset(txt_file=os.path.join(args.datadir, "identity_CelebA.txt"), root_dir=os.path.join(args.datadir, "img_align_celeba"), transform=Augmentation(224))
    train_set, validation_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    model = SimCLR_Model(args,args.embedding_space)
    trainer = L.Trainer(max_epochs=args.epochs, num_nodes=args.nodes,  devices=args.gpus)
    trainer.fit(model, train_loader, val_loader)
        
        
if __name__ == "__main__":
    main()
