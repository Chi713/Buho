import json
from collections import defaultdict

# Paths to your JSON files
TRAIN_FILE = "train_data.json"
DEV_FILE = "dev_data.json"
TEST_FILE = "test_data.json"

def load_data(file_path):
    """Loads the JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_pos_mapping(data_files):
    """
    Creates mappings from POS tags to numerical IDs and vice versa.
    Combines POS tags from all splits (train, dev, test) to create a global mapping.
    """
    all_tags = set()
    for file in data_files:
        data = load_data(file)
        for tags in data["pos_tags"]:
            all_tags.update(tags)
    
    # Create mappings
    pos_to_id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    id_to_pos = {idx: tag for tag, idx in pos_to_id.items()}
    return pos_to_id, id_to_pos

def update_data_with_ids(data, pos_to_id):
    """
    Updates the POS tags in the dataset with their corresponding numerical IDs.
    """
    updated_data = {
        "sentences": data["sentences"],
        "pos_tags": [[pos_to_id[tag] for tag in tags] for tags in data["pos_tags"]]
    }
    return updated_data

def save_data(data, file_path):
    """Saves the updated data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved updated data to {file_path}.")

def main():
    # Load the original data
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    test_data = load_data(TEST_FILE)
    
    # Create POS mappings
    pos_to_id, id_to_pos = create_pos_mapping([TRAIN_FILE, DEV_FILE, TEST_FILE])
    
    # Print mappings for verification
    print("POS to ID mapping:", pos_to_id)
    print("ID to POS mapping:", id_to_pos)
    
    # Update the data with numerical POS IDs
    train_data_updated = update_data_with_ids(train_data, pos_to_id)
    dev_data_updated = update_data_with_ids(dev_data, pos_to_id)
    test_data_updated = update_data_with_ids(test_data, pos_to_id)
    
    # Save the updated data
    save_data(train_data_updated, "train_data_updated.json")
    save_data(dev_data_updated, "dev_data_updated.json")
    save_data(test_data_updated, "test_data_updated.json")

if __name__ == "__main__":
    main()
