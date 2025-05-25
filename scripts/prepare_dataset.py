import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def create_dataset_structure():
    """Create the dataset directory structure for Hugging Face ImageFolder format."""
    # Create main dataset directory
    dataset_dir = Path("pokemon-silhouettes")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create test directory
    test_dir = dataset_dir / "test"
    test_dir.mkdir(exist_ok=True)
    
    return dataset_dir, test_dir

def copy_images(source_dir, target_dir):
    """Copy images to the test directory, organizing by Pokemon name.
    
    Args:
        source_dir: Source directory containing silhouette images
        target_dir: Base target directory (will create test subdirectory)
    """
    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    
    # Group files by Pokemon name
    pokemon_files = {}
    for file in image_files:
        pokemon_name = file.split('-')[1].split('.')[0]
        if pokemon_name not in pokemon_files:
            pokemon_files[pokemon_name] = []
        pokemon_files[pokemon_name].append(file)
    
    # Create label mapping
    label_mapping = {i: name for i, name in enumerate(sorted(pokemon_files.keys()))}
    
    # Copy files to respective directories
    for pokemon_name, files in tqdm(pokemon_files.items(), desc="Copying images"):
        # Create Pokemon directory in test
        test_pokemon_dir = target_dir / "test" / pokemon_name
        test_pokemon_dir.mkdir(exist_ok=True)
        
        # Copy files to test directory
        for file in files:
            src_path = os.path.join(source_dir, file)
            dst_path = test_pokemon_dir / file
            shutil.copy2(src_path, dst_path)
    
    # Save label mapping
    with open(target_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

def create_metadata(dataset_dir):
    """Create metadata files for the dataset."""
    metadata = []
    test_dir = dataset_dir / "test"
    
    # Load label mapping
    with open(dataset_dir / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    # Walk through all Pokemon directories
    for pokemon_dir in test_dir.iterdir():
        if pokemon_dir.is_dir():
            pokemon_name = pokemon_dir.name
            # Get all image files for this Pokemon
            image_files = [f for f in os.listdir(pokemon_dir) if f.endswith('.png')]
            
            # Create metadata entries
            for image_file in image_files:
                dex_number = image_file.split('-')[0]
                metadata.append({
                    "file_name": str(pokemon_dir / image_file),
                    "pokemon_name": pokemon_name,
                    "dex_number": dex_number,
                    "label": list(label_mapping.keys())[list(label_mapping.values()).index(pokemon_name)]
                })
    
    # Save metadata as JSONL
    with open(dataset_dir / "metadata.jsonl", "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

def main():
    # Create dataset structure
    dataset_dir, test_dir = create_dataset_structure()
    
    # Copy silhouette images
    print("Copying silhouette images...")
    copy_images("indigo-league/silhouettes", dataset_dir)
    
    # Create metadata
    print("Creating metadata...")
    create_metadata(dataset_dir)
    
    print(f"\nDataset prepared in {dataset_dir}")
    print("The dataset follows the ImageFolder format with Pokemon names as labels.")
    print("A label_mapping.json file has been created to map numeric labels to Pokemon names.")
    print("You can now load it using:")
    print(">>> from datasets import load_dataset")
    print(">>> dataset = load_dataset('imagefolder', data_dir='pokemon-silhouettes', split='test')")

if __name__ == "__main__":
    main() 