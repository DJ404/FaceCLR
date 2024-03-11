import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class FacesCombined(Dataset):
    """combines the CelebA dataset and the FFHQ dataset to a single dataset.

    Attributes:
        root_dir: directory of all the images from both datasets.
        transform: applies any data transformation. Defaults to None.
        image_paths: contains the paths to all the images.
    """
    
    
    def __init__(self, root_dir, transform=None):
        """Creates an instance of the dataset class.

        Args:
            root_dir (str): folder of all the images from both datasets.
            transform (class, optional): applies any data transformation to the images. Defaults to None.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg')) + glob.glob(os.path.join(root_dir, '*.png'))


    def __len__(self):
        """Calculates the length of the dataset.

        Returns:
            int: the length of the dataset.
        """
        
        return len(self.image_paths)


    def __getitem__(self, idx):
        """returns an image given an index from the dataset.

        Args:
            idx (int): index ranges between 0 and max length of the dataset.

        Returns:
            PIL Image object: Image at the particular index. The image is a PIL image object.
        """
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image