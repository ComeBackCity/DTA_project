from transformers import AutoTokenizer, AutoModel
import torch

def load_esm2_model(model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Load the ESM-2 model and tokenizer from Hugging Face.
    
    Args:
    - model_name (str): The name of the pre-trained ESM-2 model (default is "facebook/esm2_t33_650M_UR50D").
    
    Returns:
    - tokenizer (AutoTokenizer): The tokenizer for the ESM-2 model.
    - model (AutoModel): The pre-trained ESM-2 model.
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

def get_esm2_embeddings(sequence, tokenizer, model, device):
    """
    Tokenize a sequence and return token-wise embeddings using the ESM-2 model.
    
    Args:
    - sequence (str): The input sequence to be tokenized and converted to embeddings.
    - tokenizer (AutoTokenizer): The tokenizer for the ESM-2 model.
    - model (AutoModel): The pre-trained ESM-2 model.
    - device (torch.device): The device (GPU or CPU) used for computation.
    
    Returns:
    - token_embeddings (torch.Tensor): The token-wise embeddings for the input sequence.
    """
    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    
    # Move the inputs to the GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval()
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
    
    return token_embeddings.squeeze()[1:-1, :].cpu().detach()

# # Example usage
# tokenizer, model, device = load_esm2_model()
# sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEQLSVIGQGCFGAQNTDEKAVKKYDAKDVAAIFEDRTKRGAKLIGEIV"
# embeddings = get_esm2_embeddings(sequence, tokenizer, model, device)
# print(embeddings)

# print(len(sequence))
# print(embeddings.shape)
