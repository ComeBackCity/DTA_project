import requests
import os

def get_pdb_ids_from_uniprot(uniprot_id):
    """Retrieve PDB IDs associated with a given UniProt ID using the UniProt API."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for UniProt ID: {uniprot_id}")
        return []

    pdb_ids = []
    xml_content = response.text
    if "<dbReference type=\"PDB\" id=\"" in xml_content:
        xml_lines = xml_content.split("\n")
        for line in xml_lines:
            if "dbReference type=\"PDB\" id=\"" in line:
                start = line.find("id=\"") + 4
                end = line.find("\"", start)
                pdb_id = line[start:end]
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

def process_uniprot_ids(uniprot_ids, base_output_dir):
    """Process an array of UniProt IDs, create directories, and download PDB files."""
    for uniprot_id in uniprot_ids:
        output_dir = os.path.join(base_output_dir, uniprot_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pdb_ids = get_pdb_ids_from_uniprot(uniprot_id)
        if not pdb_ids:
            print(f"No PDB IDs found for UniProt ID: {uniprot_id}")
            continue

        for pdb_id in pdb_ids:
            download_pdb_file(pdb_id, output_dir)

def main():
    uniprot_ids = ['P22612']  # Replace with your array of UniProt IDs
    base_output_dir = "pdb_files"  # Base directory to save downloaded PDB files

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    process_uniprot_ids(uniprot_ids, base_output_dir)

if __name__ == "__main__":
    main()
