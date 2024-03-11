import argparse

import torch, torch.nn as nn
import lightning as L
import torchvision
from torchvision.datasets import LFWPeople
import torchvision.transforms as transforms

from linearmodel_pl import LinearModel
from simclr_model import SimCLR_Model

from lfw_n_classes import LFWPeopleNClasses


def main():
    """main script for running the linear evaluation.
    
    The encoder model and the dataset for perfomring linear evaluation is defined here.
    After training the linear model, it will be evaluated on a specified test set.
    """
    
    parser = argparse.ArgumentParser(description="Face Embeddings with Contrastive Learning")
    
    parser.add_argument("-b","--batch-size",  type=int, required=True, help="Number of samples in each mini-batch")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Specify starting learning rate")
    parser.add_argument("-wd", "--weight-decay", default=1e-3, type=float, help="Specify weight decay")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of total epochs")
    parser.add_argument("--embedding_space", type=int, default=512, help="The size of the embedding vector")
    parser.add_argument("-n", "--num_classes", type=int, default=10, help="Number of classes to classify")
    parser.add_argument("--emb_model", type=str, default="./models/CelebA_512_200e_007.ckpt", help="path to trained model")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes for DDP")
    
    args = parser.parse_args()
    
    args.gpus = max(torch.cuda.device_count(), 1)
    
    dataset = LFWPeopleNClasses("./data", 10, "10fold", transforms.ToTensor())
    generator = torch.Generator().manual_seed(42)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator)


    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    

    e_model = SimCLR_Model.load_from_checkpoint(args.emb_model, args=args, n_classes=args.embedding_space)
    e_model.freeze()
    
    args.num_classes = dataset.classes
    
    print(f"Classifying over {len(train_set)} + {len(validation_set)} + {len(test_set)} samples.")

    model = LinearModel(args, e_model)
    trainer = L.Trainer(max_epochs=args.epochs)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()
    