import os
from collections import defaultdict

# --- Configuration ---
# IMPORTANT: Update this list with the paths/names of your actual RSF files
ALL_RSF_FILES = [" dataset/bash/bash-dependency.rsf"]

OUTPUT_RSF_FILE = "weighted_dependencies.rsf"
ALIAS_KEY_FILE = "alias_key.txt"

# --- Global Data Structures for Mapping ---
# item_to_alias: Maps Original Name -> Short Code (e.g., 'alias.c' -> 'A2')
item_to_alias = {}
# dependency_counts: Stores the frequency of each unique, aliased dependency
dependency_counts = defaultdict(int) 
alias_counter = 0

# ----------------------------------------------------------------------
## Core Mapping Logic
# ----------------------------------------------------------------------

def get_alias(item_name: str) -> str:
    """
    Creates or retrieves a unique short code (alias) for a given item name.
    """
    global alias_counter
    # Check if the item already has an alias
    if item_name not in item_to_alias:
        # Generate a new alias (e.g., A1, A2, A3...)
        alias_counter += 1
        new_alias = f"A{alias_counter}"
        item_to_alias[item_name] = new_alias
    return item_to_alias[item_name]


def process_rsf_files(file_paths: list):
    """
    Reads all RSF files, generates aliases, and counts the frequency of each aliased dependency.
    """
    print("Starting dependency mapping and aliasing...")
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}. Skipping.")
            continue
            
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    # Expecting at least 3 parts: [RELATION_TYPE, SOURCE, TARGET]
                    if len(parts) >= 3:
                        dep_type = parts[0]
                        source_item = parts[1]
                        target_item = parts[2]

                        # Kısaltma (Aliasing)
                        aliased_source = get_alias(source_item)
                        aliased_target = get_alias(target_item)

                        # Store the aliased dependency key and increment its count
                        # Key format: (TYPE, ALIAS_SOURCE, ALIAS_TARGET)
                        aliased_dependency = (dep_type, aliased_source, aliased_target)
                        dependency_counts[aliased_dependency] += 1
                        
        except Exception as e:
            print(f"Error reading file ({path}): {e}")
    print(f"Processing complete. Found {len(dependency_counts)} unique aliased dependencies.")


# ----------------------------------------------------------------------
## Output Generation
# ----------------------------------------------------------------------

def create_weighted_rsf(output_path: str):
    """
    Writes the frequency-counted, aliased dependencies to a new RSF file.
    Format: TYPE ALIAS_SOURCE ALIAS_TARGET FREQUENCY
    """
    print(f"Creating weighted RSF file: {output_path}")
    with open(output_path, 'w') as out_f:
        for (dep_type, source_alias, target_alias), frequency in dependency_counts.items():
            out_f.write(f"{dep_type} {source_alias} {target_alias} {frequency}\n")
    print(f"Successfully created {output_path}.")


def create_alias_key_file(key_path: str):
    """
    Creates a key file mapping short codes back to their original names for LLM interpretation.
    """
    print(f"Creating alias key file: {key_path}")
    with open(key_path, 'w') as key_f:
        key_f.write("--- ALIAS KEY ---\n")
        key_f.write("SHORT_CODE: ORIGINAL_NAME\n")
        
        # Iterate over the original mapping (Original Name -> Short Code) and reverse it for the file
        # The key file should map the short code (which the LLM sees) to the original name.
        
        # We need to reverse the item_to_alias dictionary for a clear key file
        alias_to_item = {v: k for k, v in item_to_alias.items()}
        
        for alias, original in sorted(alias_to_item.items()):
             key_f.write(f"{alias}: {original}\n")
    print(f"Successfully created {key_path}.")


# ----------------------------------------------------------------------
## Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Process all input RSF files
    process_rsf_files(ALL_RSF_FILES)

    # 2. Generate the two output files needed for LLM clustering
    if dependency_counts:
        create_weighted_rsf(OUTPUT_RSF_FILE)
        create_alias_key_file(ALIAS_KEY_FILE)
        print("\n✅ Mapper completed. You can now use the two output files with CodeLlama.")
    else:
        print("\n❌ No dependencies were processed. Check your input file paths.")