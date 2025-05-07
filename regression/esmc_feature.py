from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from transformers import AutoTokenizer, AutoModel

def load_model(model_type="esm2", model_name=None):
    """
    Load either ESM2 or ESMC model based on the specified type.
    
    Args:
    - model_type (str): Either "esm2" or "esmc" to specify which model to load
    - model_name (str): Optional model name. If None, uses default for each type
    
    Returns:
    - model components: Either (tokenizer, model, device) for ESM2 or (client, device) for ESMC
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type.lower() == "esm2":
        if model_name is None:
            model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        return tokenizer, model, device
    
    elif model_type.lower() == "esmc":
        if model_name is None:
            model_name = "esmc_600m"
        client = ESMC.from_pretrained(model_name).to(device)
        return client, device
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be either 'esm2' or 'esmc'")

def get_embeddings(sequence, model_components, model_type="esm2"):
    """
    Get embeddings for a protein sequence using either ESM2 or ESMC model.
    
    Args:
    - sequence (str): The input protein sequence
    - model_components: Either (tokenizer, model, device) for ESM2 or (client, device) for ESMC
    - model_type (str): Either "esm2" or "esmc" to specify which model to use
    
    Returns:
    - embeddings (torch.Tensor): The embeddings for the input sequence
    - full_embeddings (torch.Tensor): The full embeddings including special tokens
    """
    if model_type.lower() == "esm2":
        tokenizer, model, device = model_components
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
        
        return token_embeddings.squeeze()[1:-1, :].cpu().detach(), \
               token_embeddings.squeeze().cpu().detach()
    
    elif model_type.lower() == "esmc":
        client, device = model_components
        # print(len(sequence))
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        
        logits_output = client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        
        embeddings = logits_output.embeddings.cpu().detach()
        return embeddings.squeeze()[1:-1, :].cpu().detach(), \
               embeddings.squeeze().cpu().detach()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be either 'esm2' or 'esmc'")

# # Example usage
# # For ESM2
# model_components = load_model("esm2")
# sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEQLSVIGQGCFGAQNTDEKAVKKYDAKDVAAIFEDRTKRGAKLIGEIV"
# embeddings, full_embeddings = get_embeddings(sequence, model_components, "esm2")
# print(f"ESM2 Embeddings shape: {embeddings.shape}")

# # For ESMC
# model_components = load_model("esmc")
# embeddings, full_embeddings = get_embeddings(sequence, model_components, "esmc")
# print(f"ESMC Embeddings shape: {embeddings.shape}") 