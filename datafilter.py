import json
import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgarDatasetFilter:
    def __init__(self, main_dataset_path):
        self.main_path = Path(main_dataset_path)
        self.dataset_folder = self.main_path / "dataset"
        self.images_folder = self.main_path / "images"
        self.labels_folder = self.main_path / "labels"
        self.training_lists_folder = self.main_path / "training_lists"
        self.agar_id_file = self.main_path / "AGAR_id_list.md"
        
        # Output paths for filtered dataset
        self.output_path = self.main_path / "neurosys_agar_filtered"
        self.output_images = self.output_path / "images"
        self.output_dataset = self.output_path / "dataset"
        self.output_labels = self.output_path / "labels"
        
        # Statistics
        self.stats = {
            'total_json_files': 0,
            'agar_entries_found': 0,
            'images_copied': 0,
            'missing_images': 0,
            'missing_labels': 0,
            'cleaned_entries': 0
        }
    
    def create_output_structure(self):
        """Create output directory structure"""
        self.output_path.mkdir(exist_ok=True)
        self.output_images.mkdir(exist_ok=True)
        (self.output_images / "train").mkdir(exist_ok=True)
        (self.output_images / "val").mkdir(exist_ok=True)
        self.output_dataset.mkdir(exist_ok=True)
        self.output_labels.mkdir(exist_ok=True)
        (self.output_labels / "train").mkdir(exist_ok=True)
        (self.output_labels / "val").mkdir(exist_ok=True)
        
        logger.info("Created output directory structure")
    
    def load_agar_ids(self):
        """Load AGAR IDs from the markdown file"""
        agar_ids = set()
        try:
            if self.agar_id_file.exists():
                with open(self.agar_id_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract IDs from markdown content
                    # Assuming IDs are listed in the markdown file
                    lines = content.split('\n')
                    for line in lines:
                        # Look for patterns that might be IDs
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-'):
                            # Extract potential IDs (adjust pattern as needed)
                            if line.replace('-', '').replace('_', '').isalnum():
                                agar_ids.add(line)
                
                logger.info(f"Loaded {len(agar_ids)} AGAR IDs from markdown file")
            else:
                logger.warning("AGAR_id_list.md not found, will search for 'agar' keyword in JSON files")
        except Exception as e:
            logger.error(f"Error loading AGAR IDs: {e}")
        
        return agar_ids
    
    def is_agar_dataset(self, json_data):
        """Check if the dataset is the AGAR dataset"""
        json_str = json.dumps(json_data).lower()
        
        # Check for AGAR dataset identifiers
        agar_identifiers = [
            'agar', 'microbial colony', 'neurosys', 'bacterial colony',
            'agar.neurosys.com', 'colony dataset'
        ]
        
        return any(identifier in json_str for identifier in agar_identifiers)
    
    def process_json_files(self):
        """Process all JSON files and filter agar entries"""
        agar_entries = []
        
        # First, check if annotations.json exists and is the AGAR dataset
        annotations_file = self.dataset_folder / "annotations.json"
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Check if this is the AGAR dataset
                if self.is_agar_dataset(annotations):
                    logger.info("Found AGAR dataset in annotations.json")
                    
                    # Extract all images from the AGAR dataset
                    if 'images' in annotations:
                        for image_info in annotations['images']:
                            agar_entries.append({
                                'image_info': image_info,
                                'dataset_info': annotations.get('info', {}),
                                'source_file': 'annotations.json'
                            })
                            self.stats['agar_entries_found'] += 1
                    
                    # Save complete annotations file
                    with open(self.output_dataset / "annotations.json", 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=2)
                    
                    logger.info(f"Found {len(annotations.get('images', []))} AGAR images in annotations.json")
                    return agar_entries
                
            except Exception as e:
                logger.error(f"Error processing annotations.json: {e}")
        
        # If annotations.json doesn't contain AGAR data, check individual JSON files
        logger.info("Checking individual JSON files...")
        agar_ids = self.load_agar_ids()
        
        # Check for individual JSON files (1.json to 18000.json)
        json_files_found = 0
        for i in range(1, 18001):
            json_file = self.dataset_folder / f"{i}.json"
            if json_file.exists():
                json_files_found += 1
                self.stats['total_json_files'] += 1
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if self.is_agar_dataset(data):
                        data['source_file'] = f"{i}.json"
                        agar_entries.append(data)
                        self.stats['agar_entries_found'] += 1
                        
                        if self.stats['agar_entries_found'] % 100 == 0:
                            logger.info(f"Found {self.stats['agar_entries_found']} agar entries so far...")
                
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
        
        logger.info(f"Processed {json_files_found} individual JSON files")
        return agar_entries
    
    def find_image_file(self, image_name):
        """Find image file in train or val folder"""
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for split in ['train', 'val']:
            for ext in extensions:
                # Try with original name
                img_path = self.images_folder / split / f"{image_name}{ext}"
                if img_path.exists():
                    return img_path, split
                
                # Try without extension (in case it's already included)
                img_path = self.images_folder / split / image_name
                if img_path.exists():
                    return img_path, split
        
        return None, None
    
    def copy_images_and_labels(self, agar_entries):
        """Copy corresponding images and labels for agar entries"""
        for entry in agar_entries:
            # Extract image information from JSON entry
            image_info = self.extract_image_info(entry)
            
            if not image_info:
                continue
            
            for img_name in image_info:
                # Find the actual image file
                img_path, split = self.find_image_file(img_name)
                
                if img_path:
                    # Copy image
                    dest_img_path = self.output_images / split / img_path.name
                    try:
                        shutil.copy2(img_path, dest_img_path)
                        self.stats['images_copied'] += 1
                        
                        # Copy corresponding label if it exists
                        label_name = img_path.stem + '.txt'
                        label_path = self.labels_folder / split / label_name
                        
                        if label_path.exists():
                            dest_label_path = self.output_labels / split / label_name
                            shutil.copy2(label_path, dest_label_path)
                        else:
                            self.stats['missing_labels'] += 1
                            logger.warning(f"Missing label for image: {img_name}")
                    
                    except Exception as e:
                        logger.error(f"Error copying {img_path}: {e}")
                else:
                    self.stats['missing_images'] += 1
                    logger.warning(f"Missing image: {img_name}")
    
    def extract_image_info(self, json_entry):
        """Extract image names/paths from JSON entry"""
        image_names = []
        
        # Handle COCO-style annotation format
        if 'image_info' in json_entry:
            image_info = json_entry['image_info']
            if 'file_name' in image_info:
                image_names.append(image_info['file_name'])
            return image_names
        
        # Handle other formats
        image_keys = ['image', 'images', 'file_name', 'filename', 'image_path', 'path']
        
        def search_for_images(obj, keys_found=None):
            if keys_found is None:
                keys_found = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(img_key in key.lower() for img_key in image_keys):
                        if isinstance(value, str):
                            image_names.append(value)
                        elif isinstance(value, list):
                            image_names.extend([v for v in value if isinstance(v, str)])
                    elif isinstance(value, (dict, list)):
                        search_for_images(value, keys_found + [key])
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        search_for_images(item, keys_found)
        
        search_for_images(json_entry)
        
        # Clean the image names
        cleaned_names = []
        for name in image_names:
            # Extract just the filename without path
            clean_name = os.path.basename(name)
            cleaned_names.append(clean_name)
        
        return cleaned_names
    
    def clean_dataset(self, agar_entries):
        """Remove entries with missing images or labels"""
        cleaned_entries = []
        
        for entry in agar_entries:
            image_info = self.extract_image_info(entry)
            has_all_files = True
            
            for img_name in image_info:
                img_path, split = self.find_image_file(img_name)
                if not img_path:
                    has_all_files = False
                    break
                
                # Check if corresponding label exists (optional)
                label_name = os.path.splitext(img_path.name)[0] + '.txt'
                label_path = self.labels_folder / split / label_name
                # Uncomment next lines if you want to require labels
                # if not label_path.exists():
                #     has_all_files = False
                #     break
            
            if has_all_files:
                cleaned_entries.append(entry)
                self.stats['cleaned_entries'] += 1
        
        return cleaned_entries
    
    def save_filtered_dataset(self, cleaned_entries):
        """Save the cleaned and filtered dataset"""
        # Save individual JSON files
        for i, entry in enumerate(cleaned_entries, 1):
            output_file = self.output_dataset / f"agar_{i}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, indent=2)
        
        # Save summary file
        summary = {
            'total_entries': len(cleaned_entries),
            'dataset_type': 'neurosys_agar_filtered',
            'creation_stats': self.stats
        }
        
        with open(self.output_dataset / "dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(cleaned_entries)} cleaned agar entries")
    
    def generate_report(self):
        """Generate a filtering report"""
        report = f"""
# Neurosys Agar Dataset Filtering Report

## Statistics:
- Total JSON files processed: {self.stats['total_json_files']}
- Agar entries found: {self.stats['agar_entries_found']}
- Images copied: {self.stats['images_copied']}
- Missing images: {self.stats['missing_images']}
- Missing labels: {self.stats['missing_labels']}
- Final cleaned entries: {self.stats['cleaned_entries']}

## Output Structure:
- Filtered dataset saved to: {self.output_path}
- Images: {self.output_images}
- JSON files: {self.output_dataset}
- Labels: {self.output_labels}

## Success Rate:
- Image availability: {(self.stats['images_copied'] / max(1, self.stats['agar_entries_found'])) * 100:.1f}%
- Dataset cleanliness: {(self.stats['cleaned_entries'] / max(1, self.stats['agar_entries_found'])) * 100:.1f}%
        """
        
        with open(self.output_path / "filtering_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
    
    def run_filtering(self):
        """Main method to run the entire filtering process"""
        logger.info("Starting neurosys agar dataset filtering...")
        
        # Create output structure
        self.create_output_structure()
        
        # Process JSON files and find agar entries
        logger.info("Processing JSON files...")
        agar_entries = self.process_json_files()
        
        if not agar_entries:
            logger.warning("No agar entries found!")
            return
        
        logger.info(f"Found {len(agar_entries)} agar entries")
        
        # Copy images and labels
        logger.info("Copying images and labels...")
        self.copy_images_and_labels(agar_entries)
        
        # Clean dataset (remove entries with missing files)
        logger.info("Cleaning dataset...")
        cleaned_entries = self.clean_dataset(agar_entries)
        
        # Save filtered dataset
        logger.info("Saving filtered dataset...")
        self.save_filtered_dataset(cleaned_entries)
        
        # Generate report
        self.generate_report()
        
        logger.info("Filtering completed successfully!")

# Usage
if __name__ == "__main__":
    # Update this path to your dataset location
    MAIN_DATASET_PATH = r"E:\Subhasis\combined_dataset"
    
    # Create and run the filter
    filter_tool = AgarDatasetFilter(MAIN_DATASET_PATH)
    filter_tool.run_filtering()