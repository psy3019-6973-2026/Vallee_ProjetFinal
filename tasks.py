import os
from pathlib import Path
from invoke import task
from airoh.containers import docker_run, docker_build, docker_archive, docker_setup

@task
def setup(c):
    """Install Python dependencies."""
    from airoh.utils import setup_env_python
    setup_env_python(c, "requirements.txt")
    print("✨ Setup complete!")

@task(
    help={"name": "Logical name of the file, as defined in the 'files' section of invoke.yaml."}
)
def import_file(c, name):
    """Download a single file from a URL using requests."""
    import requests

    files = c.config.get("files", {})
    if name not in files:
        raise ValueError(f"❌ No file config found for '{name}' in invoke.yaml.")

    entry = files[name]
    url = entry.get("url")
    output_file = entry.get("output_file")

    if not url or not output_file:
        raise ValueError(f"❌ Entry for '{name}' must define both 'url' and 'output_file'.")

    output_path = Path(output_file)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"🫧 Skipping {name}: {output_file} already exists.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)

    print(f"📥 Downloading '{name}' from {url}")

    try:
        with requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
                          allow_redirects=True, stream=True, timeout=60) as response, \
             tmp_path.open("wb") as f:
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)

        if total == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"❌ Downloaded 0 bytes for '{name}'.")

        tmp_path.replace(output_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"❌ Failed to download '{name}': {e}") from e

    print(f"✅ Downloaded {name} to {output_file} ({output_path.stat().st_size} bytes)")

@task
def fetch(c):
    """Download the EEG dataset from Figshare."""
    import_file(c, "eeg_dataset")
    import zipfile 
    zip_path = Path("source_data/eeg_data.zip")
    if zip_path.exists():
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("source_data/")  # Unzip 
        print("✅ Extraction complete.")

@task
def run_preprocessing(c): # only runs notebook 1
    """Run notebook 1: EEG preprocessing and bandpower feature extraction."""
    from airoh.utils import run_figures, ensure_dir_exist
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    run_figures(c, notebooks_dir, output_dir,
                keys=["source_data_dir", "output_data_dir"],
                pattern="1_*")  

@task(pre=[run_preprocessing])
def run_svm(c): # only runs notebook 2
    """Run notebook 2: SVM classification."""
    from airoh.utils import run_figures, ensure_dir_exist
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    run_figures(c, notebooks_dir, output_dir,
                keys=["source_data_dir", "output_data_dir"],
                pattern="2_*")  


@task(pre=[run_preprocessing, run_svm])
def run(c):
    """Run the full pipeline: preprocessing → SVM"""
    print("🎉 Full pipeline complete!")

@task
def clean(c):
    """Remove generated outputs."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.png")
    clean_folder(c, "output_data_dir", "*.csv")
    clean_folder(c, "output_data_dir", "*.h5")   # keras model weights
