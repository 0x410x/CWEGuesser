import json

def extract_cve_data(json_file_path):
    """
    Reads a local JSON file, extracts CVE descriptions, CWE IDs, and CVSS base scores.

    Args:
        json_file_path: The path to the local JSON file.

    Returns:
        A list of dictionaries, where each dictionary contains the description, CWE ID, and CVSS base score for a CVE entry, or None if an error occurs.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f: # Added encoding for potential Unicode issues
            cve_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return None
    except Exception as e: # Catching a broader range of exceptions
        print(f"An unexpected error occurred: {e}")
        return None

    cve_entries = []

    for item in cve_data.get('CVE_Items', []):
        try:
            description_data = item['cve']['description']['description_data']
            description = ""

            if isinstance(description_data, list):
                for desc_item in description_data:
                    value = desc_item.get('value')
                    if value:
                        description += value + " "

            elif isinstance(description_data, dict):
                value = description_data.get('value')
                if value:
                    description += value + " "

            if not description:
                continue

            cwe_id = None  # Initialize CWE ID
            cwe_data = item.get('cve', {}).get('problemtype', {}).get('problemtype_data')
            if isinstance(cwe_data, list) and cwe_data:
                cwe_id = cwe_data[0].get('description')  # Access the first element of the list, then use.get()
                if isinstance(cwe_id, list) and cwe_id:
                    cwe_id = cwe_id[0].get('value')


            cvss_base_score = None  # Initialize CVSS base score
            cvss_v3 = item.get('impact', {}).get('baseMetricV3', {}).get('cvssV3') # Handle missing data
            if cvss_v3:
                cvss_base_score = cvss_v3.get('baseScore')

            cve_entry = {
                'description': description.strip(),
                'cwe_id': cwe_id,
                'cvss_base_score': cvss_base_score
            }
            cve_entries.append(cve_entry)

        except (KeyError, TypeError) as e:
            print(f"Warning: Missing or incorrect data in a CVE entry. Skipping. Error: {e}")
            continue

    return cve_entries


# Example usage:
json_file = 'nvdcve-1.1-2024.json'  # Replace with your JSON file path
cve_data = extract_cve_data(json_file)
output_file = 'PreProcessCVEs.json'
if cve_data:
    print(f"Extracted data for {len(cve_data)} CVE entries.")

    # Write the extracted data to a new JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(cve_data, outfile, indent=4)  # Use indent for pretty printing
        print(f"Extracted data saved to {output_file}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
else:
    print("Failed to extract CVE data.")