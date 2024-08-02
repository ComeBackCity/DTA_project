import os
import requests
import xml.etree.ElementTree as ET
from Bio import SeqIO

def search_peptide_in_uniprot(sequence):
    """Search UniProt for a peptide sequence and return UniProt IDs of best matches."""
    url = f"https://www.uniprot.org/uniprot/?query={sequence}&format=xml"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to search UniProt for sequence: {sequence}")
        return []

    uniprot_ids = []
    xml_content = response.text
    root = ET.fromstring(xml_content)
    for entry in root.findall(".//{http://uniprot.org/uniprot}entry"):
        uniprot_id = entry.find("{http://uniprot.org/uniprot}accession").text
        uniprot_ids.append(uniprot_id)
    
    return uniprot_ids

def get_pdb_ids_from_uniprot(uniprot_id):
    """Retrieve PDB IDs associated with a given UniProt ID using the UniProt API."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for UniProt ID: {uniprot_id}")
        return []

    pdb_ids = []
    xml_content = response.text
    root = ET.fromstring(xml_content)
    for dbReference in root.findall(".//{http://uniprot.org/uniprot}dbReference"):
        if dbReference.attrib.get('type') == 'PDB':
            pdb_id = dbReference.attrib.get('id')
            pdb_ids.append(pdb_id)
    
    return pdb_ids

def download_pdb_file(pdb_id, output_dir):
    """Download a PDB file given a PDB ID."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded PDB file: {output_path}")
    else:
        print(f"Failed to download PDB file for PDB ID: {pdb_id}")

def main():
    fasta_file = "regression/data/fastas/davis/sequence_358.fasta"  # Replace with your FASTA file path
    output_dir = "pdb_files"  # Directory to save downloaded PDB files
    
    # Step 1: Read sequence from FASTA file
    sequence = None
    with open(fasta_file, "r") as file:
        record = SeqIO.read(file, "fasta")
        sequence = str(record.seq)
        
    if not sequence:
        print("Failed to read sequence from file.")
        return

    # Step 2: Search for UniProt IDs using UniProt peptide search
    uniprot_ids = search_peptide_in_uniprot(sequence)
    if not uniprot_ids:
        print("No UniProt IDs found for the sequence.")
        return

    # Step 3: Retrieve PDB files for each UniProt ID
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for uniprot_id in uniprot_ids:
        pdb_ids = get_pdb_ids_from_uniprot(uniprot_id)
        if not pdb_ids:
            print(f"No PDB IDs found for UniProt ID: {uniprot_id}")
            continue
        
        for pdb_id in pdb_ids:
            download_pdb_file(pdb_id, output_dir)

if __name__ == "__main__":
    main()
