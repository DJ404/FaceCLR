import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CelebADataset(Dataset):
    """transforms the CelebA dataset to a pytorch dataset.
    
    Given the root directory containing all the images from the CelebA dataset and the text file with the corresponding labels, this class transforms the dataset to a pytorch
    dataset to be useable with the pytroch framework. It includes a flag to return a triplet (anchor, positive, negative) to be compatible with the triplet loss.

    Attributes:
        data: txt file containing all the image numbers and their corresponding label.
        root_dir: directory of all the images from the dataset.
        transform: applies data transformations to the images.
        triplet: a flag, that can be set to return a triplet.
    """
    
    def __init__(self, txt_file, root_dir, transform=None, triplet=False):
        """Creates an instance of the CelebA dataset.

        Args:
            txt_file (str): path to the label text file. 
            root_dir (str): path to the folder containing all the images of the dataset.
            transform (class, optional): applies data transformations to the images. Defaults to None.
            triplet (bool, optional): If True, the class returns a triplet (anchor, positive, negative). Defaults to False.
        """
        
        self.data = pd.read_csv(txt_file, sep=" ", header=None)
        self.data[1] = self.data[1] - 1
        self.root_dir = root_dir
        self.transform = transform
        self.triplet = triplet
        
        
    def __len__(self):
        """calculates the length of the dataset.

        Returns:
            (int): the total number of images in the dataset.
        """
        
        return len(self.data)
    
    
    def __getitem__(self, index):
        """returns either an image or a tuple (if flag set) from the dataset given an index.
        
        Args:
            index (int): index between 0 and max length of the dataset.

        Returns:
            (PIL Image)/(PIL Image, PIL Image, PIL Image): image at given index. If flag is set, returns a triplet, where the positive is from the same class and negative is sampled randomly.
        """
        
        data_tuple = self.data.iloc[index]
        data_label = data_tuple[1]
        
        image_path = os.path.join(self.root_dir, data_tuple[0])
        
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
            
        if self.triplet:
            same_class = self.data[self.data[1] == data_label]
            if len(same_class) > 1:
                positive_list = same_class[same_class[0] != data_tuple[0]]
            else:
                positive_list = same_class
                
            negative_list = self.data[self.data[1] != data_label]
            
            positive_tuple = positive_list.sample()
            negative_tuple = negative_list.sample()
        
            #Generate paths for
            positive_path = os.path.join(self.root_dir, positive_tuple.iloc[0][0])
            negative_path = os.path.join(self.root_dir, negative_tuple.iloc[0][0])
        
            positive = Image.open(positive_path)
            negative = Image.open(negative_path)
        
            if self.transform:
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            return (image, positive, negative)
            
        return image
    
    # def create_dataloader(self, batch_size):
    #    return DataLoader(dataset=self, batch_size=batch_size, shuffle=True, drop_last=True)
        