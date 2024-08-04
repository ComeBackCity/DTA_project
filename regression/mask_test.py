import torch

def create_attention_mask(arr1, arr2):
    """
    Create a cross-attention mask where each element in arr1 is compared to each element in arr2.
    
    Args:
        arr1 (list): The first array.
        arr2 (list): The second array.
        
    Returns:
        torch.Tensor: The cross-attention mask.
    """
    arr1_tensor = arr1.unsqueeze(0)  # Shape: (len(arr1), 1)
    arr2_tensor = arr2.unsqueeze(1)  # Shape: (1, len(arr2))
    
    mask = (arr1_tensor == arr2_tensor).float()  # Shape: (len(arr1), len(arr2))
    
    return mask

