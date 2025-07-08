import json
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

def standardize_taxonomy_name(taxonomy_name):
    """
    Standardize a taxonomy name to medium granularity (genus and species).
    """
    if taxonomy_name is None:
        return "Unknown"
    
    # Handle generic genome category related to antimicrobial resistance
    if taxonomy_name == "Bacteria, Viruses, Fungi, and other genome sequence associated with antimicrobial resistance":
        return "Others"
    
    # Extract genus and species (usually the first two words)
    words = taxonomy_name.split()
    if len(words) >= 2:
        species_name = " ".join(words[:2])  # Keep only genus and species
        return species_name
    else:
        return taxonomy_name

def extract_data_from_json(json_file_path):
    """
    Extract relevant data from a JSON file containing CARD information.
    Returns a pandas DataFrame with taxonomy, DNA sequence, position, and drug class labels.
    """
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Temporary list to hold extracted records
    extracted_data = []
    
    # Iterate through each model in the JSON data
    for model_id, model_info in data.items():
        drug_classes = []

        # Extract drug class names from ARO_category
        if "ARO_category" in model_info:
            for category_id, category_info in model_info["ARO_category"].items():
                if category_info.get("category_aro_class_name") == "Drug Class":
                    drug_class = category_info.get("category_aro_name")
                    drug_classes.append(drug_class)
        
        # Extract sequence data if available
        if "model_sequences" in model_info and "sequence" in model_info["model_sequences"]:
            for seq_id, seq_info in model_info["model_sequences"]["sequence"].items():
                # Extract NCBI taxonomy name
                ncbi_taxonomy_name = None
                if "NCBI_taxonomy" in seq_info and "NCBI_taxonomy_name" in seq_info["NCBI_taxonomy"]:
                    full_taxonomy_name = seq_info["NCBI_taxonomy"]["NCBI_taxonomy_name"]
                    ncbi_taxonomy_name = standardize_taxonomy_name(full_taxonomy_name)
                
                # Extract DNA sequence data
                if "dna_sequence" in seq_info:
                    dna_seq_info = seq_info["dna_sequence"]
                    
                    extracted_data.append({
                        "model_id": model_id,
                        "NCBI_taxonomy_name": ncbi_taxonomy_name,
                        "fmin": dna_seq_info.get("fmin"),
                        "fmax": dna_seq_info.get("fmax"),
                        "dna_sequence": dna_seq_info.get("sequence"),
                        "drug_classes": drug_classes
                    })
    
    # Create base DataFrame (excluding drug_classes for now)
    base_df = pd.DataFrame([{k: v for k, v in item.items() if k != 'drug_classes'} for item in extracted_data])
    
    # One-hot encode drug classes using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    drug_classes_matrix = mlb.fit_transform([item['drug_classes'] for item in extracted_data])
    drug_classes_df = pd.DataFrame(drug_classes_matrix, columns=mlb.classes_)
    
    # Combine base data with one-hot encoded drug class columns
    result_df = pd.concat([base_df, drug_classes_df], axis=1)

    return result_df

# Path to the JSON file
json_file_path = "./data/raw/card.json"  # Change to your actual file path

try:
    # Extract data and generate DataFrame
    result_df = extract_data_from_json(json_file_path)
    
    # Print the result
    print(result_df)
    
    # Save to CSV
    result_df.to_csv("./data/processed/extracted_data.csv", index=False)

except Exception as e:
    print(f"Error: {e}")