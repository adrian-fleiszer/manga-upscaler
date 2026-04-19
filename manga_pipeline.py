import os
import subprocess
import shutil
import zipfile
import sys
import re

# --- CONFIGURATION SECTION ---
LOCAL_WORK_DIR = r"/home/adrian/for-extraction"
NAS_KOMGA_DIR = r"/mnt/media/manga"
UPSCALE_TOOL_DIR = r"/home/adrian/tools/manga_upscaler"
UPSCALE_TOOL_PATH = "manga_upscale.py"
HISTORY_FILE = os.path.join(UPSCALE_TOOL_DIR, "downloaded_ids.txt")

# Dynamically find the kobodl executable inside your virtual environment
KOBODL_PATH = os.path.join(os.path.dirname(sys.executable), "kobodl")

# Upscale Settings
MODEL_NAME = "2x-AnimeSharpV3.pth"
# -----------------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        for _id in history:
            f.write(f"{_id}\n")

def fetch_new_manga():
    print("--- Step 1: Checking Kobo for new books ---")
    history = load_history()
    
    list_cmd = [KOBODL_PATH, "book", "list"]
    result = subprocess.run(list_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error checking Kobo: {result.stderr}")
        return 0

    lines = result.stdout.splitlines()
    uuid_pattern = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
    new_downloads = 0
    
    for line in lines:
        if "AUDIOBOOK" in line.upper():
            continue
            
        match = uuid_pattern.search(line)
        if match:
            book_id = match.group(0)
            
            if book_id not in history:
                print(f"Checking availability for ID: {book_id}...")
                
                # Download directly into the local extraction folder
                get_cmd = [KOBODL_PATH, "book", "get", "--format-str", "{Title}", book_id]
                get_result = subprocess.run(get_cmd, capture_output=True, text=True, cwd=LOCAL_WORK_DIR)
                
                if get_result.returncode != 0:
                    # If it fails, we DO NOT add it to history. 
                    # We just print a clean message and move on so it tries again tomorrow.
                    print(f"  -> Skipping: Not available for download yet (Pre-order / Pending release).")
                    continue
                
                # If we make it here, the book actually downloaded!
                print(f"  -> Success! Downloaded new release.")
                history.add(book_id)
                new_downloads += 1
                save_history(history)          
      
    if new_downloads > 0:
        print(f"Downloaded {new_downloads} new files. Formatting as .cbz...")
        for root, dirs, files in os.walk(LOCAL_WORK_DIR):
            for filename in files:
                if filename.endswith(".epub"):
                    old_path = os.path.join(root, filename)
                    base = os.path.splitext(filename)[0]
                    new_path = os.path.join(LOCAL_WORK_DIR, base + ".cbz")
                    os.rename(old_path, new_path)
                    print(f"Staged: {base}.cbz")
        
        # Cleanup the nested kobo_downloads directory if it was created
        kobo_dir = os.path.join(LOCAL_WORK_DIR, "kobo_downloads")
        if os.path.exists(kobo_dir):
            shutil.rmtree(kobo_dir)
            
    return new_downloads

def create_cbz_from_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def find_main_image_directory(base_path):
    max_images = 0
    best_dir = None
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    for root, dirs, files in os.walk(base_path):
        image_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in image_extensions)
        if image_count > max_images:
            max_images = image_count
            best_dir = root
    return best_dir

def process_manga():
    print(f"\n--- Step 2: Extracting and Upscaling ---")
    extract_cmd = [sys.executable, UPSCALE_TOOL_PATH, "extract", "--input", LOCAL_WORK_DIR]
    subprocess.run(extract_cmd, check=True)

    items = os.listdir(LOCAL_WORK_DIR)
    for item in items:
        current_dir_path = os.path.join(LOCAL_WORK_DIR, item)
        if not os.path.isdir(current_dir_path):
            continue

        print(f"\nProcessing: {item}")
        image_dir = find_main_image_directory(current_dir_path)
        if not image_dir:
            continue

        output_dir = os.path.join(os.path.dirname(image_dir), "upscaled_temp")
        
        upscale_cmd = [
            sys.executable, UPSCALE_TOOL_PATH, "upscale",
            "--color", image_dir,
            "--output", output_dir,
            "--model-color", MODEL_NAME
        ]
        
        try:
            subprocess.run(upscale_cmd, check=True)
            shutil.rmtree(image_dir)
            shutil.move(output_dir, image_dir)
            
            cbz_output_path = os.path.join(LOCAL_WORK_DIR, f"{item}.cbz")
            create_cbz_from_folder(current_dir_path, cbz_output_path)
        except Exception as e:
            print(f"Error processing {item}: {e}")
            
        shutil.rmtree(current_dir_path)

def route_to_komga():
    print("\n--- Step 3: Routing to Komga NAS ---")
    for filename in os.listdir(LOCAL_WORK_DIR):
        if filename.endswith(".cbz"):
            name_without_ext = os.path.splitext(filename)[0]
            series_name = re.sub(r'(?i)[,\s-]*(?:vol(?:ume|\.)?|v\.?)?\s*\d+.*$', '', name_without_ext).strip()
            series_dir = os.path.join(NAS_KOMGA_DIR, series_name)
            
            if not os.path.exists(series_dir):
                os.makedirs(series_dir)
                print(f"Created directory: {series_dir}")
                
            src = os.path.join(LOCAL_WORK_DIR, filename)
            dst = os.path.join(series_dir, filename)
            shutil.move(src, dst)
            print(f"Moved {filename} to NAS -> {series_name}")


if __name__ == "__main__":
    os.chdir(UPSCALE_TOOL_DIR)
    
    new_books = fetch_new_manga()
    if new_books > 0:
        process_manga()
        route_to_komga()
    else:
        print("No new books found.")
        
    print("\n--- Finished. leaving VM awake to go to sleep on its own. ---")
