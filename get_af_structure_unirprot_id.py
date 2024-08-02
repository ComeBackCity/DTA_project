import requests
import os
from tdc.multi_pred import DTI
import pandas as pd

def get_alphafold_structure(uniprot_id):
    """Retrieve the AlphaFold structure associated with a given UniProt ID."""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve AlphaFold structure for UniProt ID: {uniprot_id}")
        return None

def download_alphafold_structure(uniprot_id, output_dir):
    """Download the AlphaFold structure given a UniProt ID and save it to the output directory."""
    structure = get_alphafold_structure(uniprot_id)
    if structure:
        output_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")
        with open(output_path, 'wb') as file:
            file.write(structure)
        print(f"Downloaded AlphaFold structure: {output_path}")

def process_uniprot_ids(uniprot_ids, base_output_dir):
    """Process an array of UniProt IDs, create directories, and download AlphaFold structures."""
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    for uniprot_id in uniprot_ids:
        download_alphafold_structure(uniprot_id, base_output_dir)
        
def get_uniprot_ids(dataset):
    data = DTI(name = dataset)
    split = data.get_split()
    return pd.unique(split['train']['Target_ID'])

def get_toxcast_ids():
    df = pd.read_csv("regression/data/full_toxcast/raw/data.csv")
    ids = pd.unique(df['Uniprot_ID'])
    return ids

def main():
    # from graphein.protein.utils import download_alphafold_structure

    kiba_ids = get_uniprot_ids('KIBA')
    # binding_db_ids = get_uniprot_ids('Binding_DB')
    toxcast_ids = get_toxcast_ids()
    base_output_dir = "alphafold_structures"  # Base directory to save downloaded structures

    process_uniprot_ids(kiba_ids, os.path.join(base_output_dir, "kiba"))
    # process_uniprot_ids(binding_db_ids, os.path.join(base_output_dir, "binding_db"))
    process_uniprot_ids(toxcast_ids, os.path.join(base_output_dir, "toxcast"))

if __name__ == "__main__":
    main()
