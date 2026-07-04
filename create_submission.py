import os
import zipfile

def create_submission_zip():
    source_dir = "code"
    output_filename = "code.zip"
    
    # Files and folders to exclude
    exclude_dirs = {"__pycache__", "venv", ".venv", "node_modules", ".git"}
    exclude_files = {".env", output_filename}
    
    print(f"Creating {output_filename} from '{source_dir}' directory...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files or file.endswith(".pyc"):
                    continue
                    
                file_path = os.path.join(root, file)
                # Ensure the paths in the zip file are relative to the 'code' directory
                arcname = os.path.relpath(file_path, start=os.path.dirname(source_dir))
                
                print(f"Adding: {arcname}")
                zipf.write(file_path, arcname)
                
    print(f"\nSuccess! Your submission file '{output_filename}' is ready in the root folder.")

if __name__ == "__main__":
    create_submission_zip()
