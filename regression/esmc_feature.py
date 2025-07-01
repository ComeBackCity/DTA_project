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
    - residue_embeddings (torch.Tensor): The embeddings for the protein residues (excluding special tokens)
    - cls_embedding (torch.Tensor): The embedding for the CLS token (for ESM2), or None for ESMC
    """
    if model_type.lower() == "esm2":
        tokenizer, model, device = model_components
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0) # [seq_len, hidden_size]
        
        # Extract CLS token embedding (usually the first token)
        cls_embedding = token_embeddings[0, :].cpu().detach()
        
        # Extract residue embeddings (excluding CLS and SEP tokens)
        residue_embeddings = token_embeddings[1:-1, :].cpu().detach()
        
        return residue_embeddings, cls_embedding
    
    elif model_type.lower() == "esmc":
        client, device = model_components
        # print(len(sequence))
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        # print(protein_tensor)
        
        logits_output = client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )

        # print(logits_output)
        embeddings = logits_output.embeddings.squeeze(0).cpu().detach()
        # print(embeddings.shape)
        # exit()
        # ESMC might not have a distinct CLS token in the same way as BERT-like models.
        # Returning the sequence embeddings excluding the first and last tokens.
        residue_embeddings = embeddings[1:-1, :]
        cls_embedding = embeddings[0, :].unsqueeze(0)

        
        return residue_embeddings, cls_embedding
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be either 'esm2' or 'esmc'")

# # Example usage
# # For ESM2
# model_components = load_model("esm2")
# sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEQLSVIGQGCFGAQNTDEKAVKKYDAKDVAAIFEDRTKRGAKLIGEIV"
# residue_embeddings, cls_embedding = get_embeddings(sequence, model_components, "esm2")
# print(f"ESM2 Residue Embeddings shape: {residue_embeddings.shape}")
# print(f"ESM2 CLS Embedding shape: {cls_embedding.shape}")

# # For ESMC
# model_components = load_model("esmc")
# residue_embeddings, cls_embedding = get_embeddings(sequence, model_components, "esmc")
# print(f"ESMC Residue Embeddings shape: {residue_embeddings.shape}")
# if cls_embedding is not None:
#     print(f"ESMC CLS Embedding shape: {cls_embedding.shape}") 