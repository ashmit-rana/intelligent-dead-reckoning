"""
download_real_dataset.py
========================
Downloads real vehicle and smartphone sensor datasets from the IO-VNBD
GitHub repository (https://github.com/onyekpeu/IO-VNBD) by resolving
Git LFS (Large File Storage) pointers directly via GitHub's LFS Batch API.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error

REPO = "onyekpeu/IO-VNBD"
LFS_BATCH_URL = f"https://github.com/{REPO}.git/info/lfs/objects/batch"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO}"

def fetch_url(url, headers=None, data=None):
    req = urllib.request.Request(url, headers=headers or {}, data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()

def get_default_branch():
    api_url = f"https://api.github.com/repos/{REPO}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "IDR-Downloader"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("default_branch", "master")
    except Exception:
        return "master"

def get_repo_tree(branch):
    api_url = f"https://api.github.com/repos/{REPO}/git/trees/{branch}?recursive=1"
    req = urllib.request.Request(api_url, headers={"User-Agent": "IDR-Downloader"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def parse_lfs_pointer(text):
    oid_match = re.search(r"oid sha256:([a-f0-9]{64})", text)
    size_match = re.search(r"size (\d+)", text)
    if oid_match and size_match:
        return oid_match.group(1), int(size_match.group(1))
    return None, None

def resolve_lfs_download_url(oid, size):
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}]
    }).encode("utf-8")
    
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "git-lfs/3.0.0"
    }
    
    req = urllib.request.Request(LFS_BATCH_URL, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        obj = data["objects"][0]
        if "error" in obj:
            raise RuntimeError(f"LFS API error: {obj['error']}")
        download_info = obj["actions"]["download"]
        return download_info["href"], download_info.get("header", {})

def download_file(href, dest_path, headers=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(href, headers=headers or {"User-Agent": "IDR-Downloader"})
    print(f"Downloading to {dest_path}...")
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 64
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                print(f"\r  [{percent:5.1f}%] {downloaded/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB", end="", flush=True)
        print(" -> Done!")

def main():
    dest_dir = "data/raw"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"1. Querying repository {REPO}...")
    branch = get_default_branch()
    print(f"   Default branch: {branch}")
    
    tree_data = get_repo_tree(branch)
    tree = tree_data.get("tree", [])
    
    csv_files = [f for f in tree if f["path"].endswith(".csv")]
    print(f"   Found {len(csv_files)} CSV files in repository.")
    
    # We want S-S1.csv (smartphone) and V-S1.csv (vehicle ground truth) or similar drives
    target_stems = ["S-S1.csv", "V-S1.csv", "S-S2.csv", "V-S2.csv"]
    targets_to_download = []
    
    for f in csv_files:
        path = f["path"]
        for stem in target_stems:
            if path.endswith(stem):
                targets_to_download.append((stem, path))
                break
                
    if not targets_to_download:
        # Fall back to first 2 CSV files
        targets_to_download = [(os.path.basename(f["path"]), f["path"]) for f in csv_files[:2]]
        
    print(f"\n2. Selected files to download:")
    for stem, path in targets_to_download:
        print(f"   - {stem} (path: {path})")
        
    for stem, path in targets_to_download:
        target_dest = os.path.join(dest_dir, stem)
        if os.path.exists(target_dest) and os.path.getsize(target_dest) > 100000:
            print(f"\nFile {stem} already exists ({os.path.getsize(target_dest)/(1024*1024):.2f} MB). Skipping download.")
            continue
            
        print(f"\nFetching pointer for {path}...")
        raw_url = f"{RAW_BASE_URL}/{branch}/{urllib.parse.quote(path)}"
        raw_content = fetch_url(raw_url).decode("utf-8", errors="ignore")
        
        oid, size = parse_lfs_pointer(raw_content)
        if oid and size:
            print(f"   Git LFS Object detected: oid={oid[:12]}..., size={size/(1024*1024):.2f} MB")
            download_url, headers = resolve_lfs_download_url(oid, size)
            download_file(download_url, target_dest, headers)
        else:
            print(f"   Normal file (not LFS). Saving directly...")
            with open(target_dest, "w", encoding="utf-8") as out:
                out.write(raw_content)

if __name__ == "__main__":
    main()
