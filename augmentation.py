import torchvision.transforms as transforms

class Augmentation:
    """Callable class to apply the chosen data augmentations to the dataset.
    
    The data augmentations are identical to the ones in the SimCLR paper and are taken from the torchvision library.
    
    Attributes:
        transform: composition of all the data augmentations.
    """
    
    def __init__(self, input_size, s=1.0):
        """Creates an instance of the augmentation class.
        
        Augmentations are depending on the given input size. Color jitter has a scalable parameter s.

        Args:
            input_size (int): the size of the input images to the network.
            s (float, optional): scales the intensity of the color jitter parameters. Defaults to 1.0.
        """
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        blur = transforms.GaussianBlur(kernel_size=int(0.1*input_size)+1, sigma=(0.1, 2.0))
        
        self.transform = transforms.Compose(
        [transforms.RandomResizedCrop(input_size), 
         transforms.RandomHorizontalFlip(), 
         transforms.RandomApply([color_jitter], p=0.8), 
         transforms.RandomGrayscale(p=0.2),
         transforms.RandomApply([blur], p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __call__(self, x):
        """applies the defined augmentations by calling the class.

        Args:
            x (tensor): input to the class.

        Returns:
            (tensor, tensor): a tuple of different augmented versions of the input.
        """
        return self.transform(x), self.transform(x)