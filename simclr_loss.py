import torch
import torch.nn.functional as F
import torch.nn as nn

class SimCLR_Loss(nn.Module):
    """implements the NT-Xent loss as a pytorch module from the SimCLR paper.
    

    Attributes:
        batch_size: the total batch_size across all gps and all nodes.
        temperature: the temperature scale used in the original paper.
        mask: matrix mask which indicates all negatives in one batch.
    """
    
    def __init__(self, batch_size, num_nodes,  gpu_count, temperature=0.07):
        """creates an instance of the NT-Xent loss.

        Args:
            batch_size (int): batch size of one gpu used to calculate the total batch size.
            num_nodes (int): number of computational nodes in a cluster used to calculate the total batch size.
            gpu_count (int): number of gpus in one computational node used to calculate the total batch size.
            temperature (float, optional): temperature scale used in the NT-Xent loss scales the values of every embedding vector. Defaults to 0.07.
        """
        super().__init__()
        self.batch_size = batch_size * num_nodes * gpu_count 
        self.temperature = temperature
        self.mask = (~torch.eye(2 * self.batch_size, 2 * self.batch_size, dtype=bool))
        
        
    def forward(self, emb_i, emb_j):
        """handles the loss calculation for the input embedding vectors.

        Args:
            emb_i (tensor): embeddings of the augmented input images.
            emb_j (tensor): correpsonding embeddings of the input images augmented differently.

        Returns:
            (tensor): NT-Xent loss of one batch of pairs of varying augmented images.
        """
        
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        
        embeddings = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        sim_i_j = torch.diag(similarity_matrix, self.batch_size)
        sim_j_i = torch.diag(similarity_matrix, -self.batch_size)
        
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        top = torch.exp(positives / self.temperature)
        
        # mask = self.mask.to(device)
        
        # bottom = mask * torch.exp(similarity_matrix / self.temperature)
        
        negatives = similarity_matrix[self.mask].reshape(2 * self.batch_size, -1).contiguous()
        
        bottom =  torch.exp(negatives/ self.temperature)
        
        loss_pair = -torch.log(top / torch.sum(bottom, dim=1))
        
        return torch.sum(loss_pair) / (2 * self.batch_size)
        
