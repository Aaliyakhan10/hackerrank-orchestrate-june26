import os
import json
import imagehash
from PIL import Image
from typing import Dict, Any, List

class RetrievalMemory:
    def __init__(self, memory_file="../dataset/memory_db.json"):
        self.memory_file = memory_file
        # Create dir if not exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.memory_db = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def _save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory_db, f, indent=2)

    def check_for_duplicates(self, claim_id: str, image_paths: List[str]) -> Dict[str, Any]:
        """
        Hashes images and checks for duplicates in memory.
        If a duplicate is found, returns a risk flag.
        Then, saves the new hashes to memory.
        """
        results = {
            "duplicate_found": False,
            "duplicate_details": [],
            "retrieval_flags": []
        }
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
                
            try:
                img = Image.open(img_path)
                # Compute perceptual hash
                phash = str(imagehash.phash(img))
                
                # Check for duplicates
                for entry in self.memory_db:
                    # Ignore the same claim if re-running tests
                    if entry["claim_id"] == claim_id:
                        continue
                        
                    stored_hash = imagehash.hex_to_hash(entry["phash"])
                    current_hash = imagehash.hex_to_hash(phash)
                    if current_hash - stored_hash <= 3: # threshold for "same image"
                        results["duplicate_found"] = True
                        results["duplicate_details"].append(f"Image matches past claim {entry['claim_id']}")
                        if "reused_fraud_photo" not in results["retrieval_flags"]:
                            results["retrieval_flags"].append("reused_fraud_photo")
                
                # Update memory
                # Prevent duplicate entries for same claim/image during re-runs
                exists = any(e["claim_id"] == claim_id and e["image_path"] == img_path for e in self.memory_db)
                if not exists:
                    self.memory_db.append({
                        "claim_id": claim_id,
                        "image_path": img_path,
                        "phash": phash
                    })
            except Exception as e:
                print(f"Error processing {img_path} in memory: {e}")
                
        self._save_memory()
        return results
