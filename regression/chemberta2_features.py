from transformers import AutoTokenizer, AutoModel
import torch

def load_ehmberta2_model(model_name="seyonec/PubChem10M_SMILES_BPE_450k"):
    """
    Load the EHM-BERTa2 model and tokenizer from Hugging Face.
    
    Args:
    - model_name (str): The name of the pre-trained EHM-BERTa2 model (default is "seyonec/PubChem10M_SMILES_BPE_450k").
    
    Returns:
    - tokenizer (AutoTokenizer): The tokenizer for the EHM-BERTa2 model.
    - model (AutoModel): The pre-trained EHM-BERTa2 model.
    - device (torch.device): The device (GPU or CPU) used for computation.
    """
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move the model to the GPU if available
    model.to(device)
    
    return tokenizer, model, device

def get_ehmberta2_embeddings(smiles_string, tokenizer, model, device):
    """
    Tokenize a SMILES string and return token-wise embeddings using the EHM-BERTa2 model.
    
    Args:
    - smiles_string (str): The input SMILES string to be tokenized and converted to embeddings.
    - tokenizer (AutoTokenizer): The tokenizer for the EHM-BERTa2 model.
    - model (AutoModel): The pre-trained EHM-BERTa2 model.
    - device (torch.device): The device (GPU or CPU) used for computation.
    
    Returns:
    - token_embeddings (torch.Tensor): The token-wise embeddings for the input SMILES string.
    """
    # Tokenize the input SMILES string
    inputs = tokenizer(smiles_string, return_tensors="pt", add_special_tokens=True)
    
    # Move the inputs to the GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval()
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
    
    # Return the full embedding including CLS token
    return token_embeddings.squeeze().cpu().detach()

# # Example usage
# tokenizer, model, device = load_ehmberta2_model()
# smiles = "CCO"  # Example SMILES string for ethanol
# embeddings = get_ehmberta2_embeddings(smiles, tokenizer, model, device)
# print(embeddings)
