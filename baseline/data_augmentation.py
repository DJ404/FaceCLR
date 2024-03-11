import torchvision.transforms as transforms

class DataAug:
    """ Callable class, where the chosen data augmentations are defined.
    
    This particular data augmentation class is specialised for supervised learning. 
    It differs from the augmentations used for contrastive learning. It uses a softer random crop.
    The data augmentations are taken from the torchvision library.
    
    Attributes:
        transform: all chosen data augmentations composed.
    """
    
    def __init__(self, input_size):
        """Creates an instance of the DataAug class with given input size.

        Args:
            input_size (int): the size of the images going into the network.
        """
        
        
        blur = transforms.GaussianBlur(kernel_size=int(0.1*input_size)+1, sigma=(0.1, 0.5))
        
        self.transform = transforms.Compose(
        [transforms.RandomResizedCrop(input_size, scale=(0.8, 1)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([blur], p=0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        transforms.Normalize((0.5,), (0.5))
        ])
        
    def __call__(self, x):
        """Calling the class after instantiating it applies the augmentations to x.

        Args:
            x (tensor): input to the class.

        Returns:
            (tensor): augmented version of the input with all the data augmentations applied.
        """
        
        
        return self.transform(x)