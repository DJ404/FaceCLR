from torchvision.datasets import LFWPeople
from torch.utils.data import Dataset

from collections import Counter

class LFWPeopleNClasses(Dataset):
    """LFW subset that contains N classes of identities with the highest sample count.
    
    Contains the LFWPeople dataset from the Pytorch library and creates a subset cotianing only N identities.

    Attributes:
        dataset: LFWPeople dataset from the torchvision package.
        classes: the number of different identities the subset will contain.
        transform: saves the transformation which is applied to the image.
        indicies: all the valid image indicies which belong to one of the N classes.
        new_labels: mapping from the old labels to new ones after filtering the dataset, so they start from 0 and are contiguous.
        targets: valid targets that belong to the N classes.
    """
    
    def __init__(self, root, n_classes, split, transform=None):
        """Creates an instance of the filtered LFW dataset.

        Args:
            root (str): the location of the root from the LFW dataset.
            n_classes (int): the final number of classes the dataset will contain.
            split (str): the split that will be used for filtering. Either one of "train", "test", "10fold".
            transform (class, optional): data augmentations that will be applied to the LFWPeople dataset class. Defaults to None.
        """
        
        self. dataset = LFWPeople(root, split=split,image_set="deepfunneled",transform=transform, download=True)
        self.classes = n_classes
        self.transform = transform
        
        counts = Counter(self.dataset.targets)
        n_classes_max_img = [class_id for class_id, count in counts.most_common(n_classes)]
        self.indices = [i for i, target in enumerate(self.dataset.targets) if target in n_classes_max_img]
        self.new_labels = {original_label: new_label for new_label, original_label in enumerate(sorted(n_classes_max_img))}
        self.targets = [self.new_labels[self.dataset.targets[i]] for i in self.indices]


    def __len__(self):
        """Calculates the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        
        return len(self.indices)
    

    def __getitem__(self, idx):
        """samples one item (image, label) from the dataset given an index value.
        
        Args:
            idx (int): the index indicates the sample which will be chosen.

        Returns:
            tuple (tensor, tensor): the tuple with the image and its corresponding label.
        """
        
        img, _ = self.dataset[self.indices[idx]]
        label = self.targets[idx]
        
        return img, label


class TransformationDataset(Dataset):
    """Wrapper for datasets to apply data transformations.

    Attributes:
        dataset: the dataset to which the transformation will be applied.
        transform: the transformation that will be applied.
    """
    
    
    def __init__(self, dataset, transform=None):
        """Creates an instance of the transformation dataset.

        Args:
            dataset (class): dataset that will be augmented.
            transform (class, optional): single or composition of transformations that will be applied. Defaults to None.
        """
        
        self.dataset = dataset
        self.transform = transform
        
        
    def __getitem__(self, index):
        """extracts a sample from the dataset given the index.

        Args:
            index (int): the index of a sample.

        Returns:
            tuple: (x,y) where x is the datapoint and y the corresponding label.
        """
        
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
        
    def __len__(self):
        """calculates the length of the dataset.

        Returns:
            int: the total number of data samples in the dataset.
        """
        
        return len(self.dataset)