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
    """Download the EEG dataset from Figshare via the API."""
    import requests

    files = c.config.get("files", {})
    entry = files.get("eeg_dataset", {})
    api_url = entry.get("url")
    output_dir = Path(entry.get("output_dir", "source_data"))

    if not api_url:
        raise ValueError("❌ No 'url' configured for 'eeg_dataset' in invoke.yaml.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Fetching file list from {api_url}")
    file_list = []
    page = 1
    while True:
        response = requests.get(api_url, params={"page": page, "page_size": 100}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        file_list.extend(batch)
        print(f"  📄 Page {page}: {len(batch)} fichiers trouvés")
        page += 1
    print(f"  ✅ Total: {len(file_list)} fichiers à télécharger")

    for file_info in file_list:
        name = file_info["name"]
        download_url = file_info["download_url"]
        output_path = output_dir / name
        tmp_path = output_path.with_suffix(output_path.suffix + ".part")

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"🫧 Skipping '{name}': already exists.")
            continue

        tmp_path.unlink(missing_ok=True)
        print(f"📥 Downloading '{name}' from {download_url}")

        try:
            with requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"},
                              allow_redirects=True, stream=True, timeout=60) as dl_response, \
                 tmp_path.open("wb") as f:
                dl_response.raise_for_status()
                total = 0
                for chunk in dl_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)

            if total == 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"❌ Downloaded 0 bytes for '{name}'.")

            tmp_path.replace(output_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"❌ Failed to download '{name}': {e}") from e

        print(f"✅ Downloaded '{name}' ({output_path.stat().st_size} bytes)")

@task
def run_preprocessing(c):
    """Run notebook 1: EEG preprocessing and bandpower feature extraction."""
    import subprocess
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    notebook = notebooks_dir / "1_preprocessing_bandpower.ipynb"
    env = os.environ.copy()
    env["SOURCE_DATA_DIR"] = str(Path(c.config.get("source_data_dir")).resolve())
    env["OUTPUT_DATA_DIR"] = str(Path(c.config.get("output_data_dir")).resolve())
    Path(c.config.get("output_data_dir")).mkdir(parents=True, exist_ok=True)
    print(f"▶️ Running {notebook.name}...")
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", str(notebook)],
        env=env, check=True
    )
    print(f"✅ {notebook.name} complete!")

@task(pre=[run_preprocessing])
def run_svm(c):
    """Run notebook 2: SVM classification."""
    import subprocess
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    notebook = notebooks_dir / "2_ml_svm.ipynb"
    env = os.environ.copy()
    env["SOURCE_DATA_DIR"] = str(Path(c.config.get("source_data_dir")).resolve())
    env["OUTPUT_DATA_DIR"] = str(Path(c.config.get("output_data_dir")).resolve())
    print(f"▶️ Running {notebook.name}...")
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", str(notebook)],
        env=env, check=True
    )
    print(f"✅ {notebook.name} complete!")


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
