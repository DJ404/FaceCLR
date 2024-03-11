import numpy as np
import argparse
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from simclr_model import SimCLR_Model
from lfw_n_classes import LFWPeopleNClasses

import torch
from torchvision import transforms

def plot_tsne(data, labels, n_classes, perplexity=None, n_iter=1000):
    """creates a TSNE plot for N classes with the most samples for given embeddings and labels.
    
    Args:
        data (ndarray): array with the embeddings.
        labels (ndarray): array with corresponding labels of the embeddings.
        n_classes (int): number of different classes to plot.
        perplexity (float, optional): number of nearest neighbors. If None, it will calculate the number depeneding on the number of samples. Defaults to None.
        n_iter (int, optional): max number of iterations. Defaults to 1000.
    """

    most_labels = Counter(labels)
    selected_labels = [label for label, _ in most_labels.most_common(n_classes)]
    mask = np.isin(labels, selected_labels)
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    
    #So it doesnt crash, when perplexity is too low, bc of too little data
    if perplexity is None:
        num_samples = len(filtered_data)
        perplexity = min(30, num_samples - 1)

    #Generate TSNE components
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(filtered_data)

    #Create plot
    plt.figure(figsize=(10, 8))
    for i in selected_labels:
        indices = filtered_labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {i}')
    plt.legend()
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
    


def generate_embeddings(data_loader, model, device='cpu'):
    """generates embeddings with a given encoder model for a given dataloader.

    Args:
        data_loader: DataLoader of the dataset for generating the embeddings.
        model: Pretrained encoder model.
        device (str, optional): Flag to specify to run on either "cpu" or "gpu". Defaults to 'cpu'.

    Returns:
        (ndarray, ndarray): tuple containing the embeddings and the correspondign labels.
    """
    
    model = model.to(device)
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (images, batch_labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating embeddings"):

            images = images.to(device)
            embeddings = model(images)

            features.extend(embeddings.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Face Embeddings with Contrastive Learning")
    
    parser.add_argument("-b","--batch-size",  type=int, required=True, help="Number of samples in each mini-batch")
    parser.add_argument("--embedding_space", type=int, default=512, help="The size of the embedding vector")
    parser.add_argument("-n", "--num_classes", type=int, default=10, help="Number of classes to classify")
    parser.add_argument("--emb_model", type=str, default="./models/CelebA_512_200e_007.ckpt", help="path to trained model")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes for DDP")
    
    args = parser.parse_args()
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.gpus = max(torch.cuda.device_count(), 1)
    args.temperature = 0.5
    
    e_model = SimCLR_Model.load_from_checkpoint(args.emb_model, args=args, n_classes=args.embedding_space)
    e_model.freeze()
    
    #dataset = CelebADataset("C:/Users/David/Documents/Python_Scripts/ContrastiveFaceEmb128/data/identity_CelebA.txt", "C:/Users/David/Documents/Python_Scripts/ContrastiveFaceEmb128/data/img_align_celeba/", transform=img_transform)
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    test_set = LFWPeopleNClasses("./data",10, "test", transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    
    features, labels = generate_embeddings(test_loader, e_model, device=dev)
    
    plot_tsne(features, labels, n_classes=10)
    
if __name__ == "__main__":
    main()