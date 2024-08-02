import requests
import os

def get_uniprot_id(sequence):
    """Retrieve the UniProt ID associated with a given protein sequence."""
    url = "https://www.uniprot.org/uniprot/?query=sequence:%22" + sequence + "%22&format=tab&columns=id"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split('\n')
        if len(lines) > 1:
            return lines[1].strip()  # Return the first UniProt ID found
    print(f"Failed to retrieve UniProt ID for sequence: {sequence}")
    return None

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
        # Create a separate directory for each UniProt ID
        uniprot_dir = os.path.join(output_dir, uniprot_id)
        if not os.path.exists(uniprot_dir):
            os.makedirs(uniprot_dir)
        
        output_path = os.path.join(uniprot_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")
        with open(output_path, 'wb') as file:
            file.write(structure)
        print(f"Downloaded AlphaFold structure: {output_path}")

def process_sequences(sequences, base_output_dir):
    """Process an array of sequences, find corresponding UniProt IDs, and download AlphaFold structures."""
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    for sequence in sequences:
        uniprot_id = get_uniprot_id(sequence)
        if uniprot_id:
            download_alphafold_structure(uniprot_id, base_output_dir)

def main():
    sequences = [
        "MSTPVRLTLFCDFVDMDDYSNQFEGFRHPSRDYPFLFYYHILPEINLRTSTQLTFFPAQEYFRQLDPEFMSRLHKHLSS",
        "MKAILVVLLYTFTPYTAYLAFTRGEL",
        "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPFNATLSELHCDKLHVDPENFRLLGNMKHLVR"
    ]  # Replace with your array of sequences
    base_output_dir = "alphafold_structures"  # Base directory to save downloaded structures

    process_sequences(sequences, base_output_dir)

if __name__ == "__main__":
    main()
