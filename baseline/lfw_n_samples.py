from torchvision.datasets import LFWPeople
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset
from data_augmentation import DataAug

from collections import Counter


class LFWPeopleNSamples(Dataset):
    def __init__(self, root, n_img, split):
        self. dataset = LFWPeople(root, split=split,image_set="deepfunneled",transform=transforms.ToTensor(), download=True)
        self.n_img = n_img
        self.indices = self._valid_indices()
        self.remapped_labels = self._new_labels()
        self.classes = max(self.remapped_labels) + 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        label = self.remapped_labels[idx]
        return img, label
    
    def _new_labels(self):
        original_labels = [self.dataset.targets[i] for i in self.indices]
        unique_original_labels = sorted(set(original_labels))
        new_label_mapping = {original_label: new_label for new_label, original_label in enumerate(unique_original_labels)}
        remapped_labels = [new_label_mapping[label] for label in original_labels]
        return remapped_labels
    
    def _valid_indices(self):
        counts = Counter(self.dataset.targets)
        valid_indices = [i for i, target in enumerate(self.dataset.targets) if counts[target] >= self.n_img]
        return valid_indices