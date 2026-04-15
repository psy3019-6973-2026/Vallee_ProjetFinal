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

@task
def fetch(c):
    """Download the EEG dataset from Figshare."""
    from tasks import import_file 
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

@task(pre=[run_preprocessing])
def run_eegnet(c): # only runs notebook 3
    """Run notebook 3: EEGNet deep learning."""
    from airoh.utils import run_figures, ensure_dir_exist
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    run_figures(c, notebooks_dir, output_dir,
                keys=["source_data_dir", "output_data_dir"],
                pattern="3_*")  

@task(pre=[run_preprocessing, run_svm, run_eegnet])
def run(c):
    """Run the full pipeline: preprocessing → SVM → EEGNet."""
    print("🎉 Full pipeline complete!")

@task
def clean(c):
    """Remove generated outputs."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.png")
    clean_folder(c, "output_data_dir", "*.csv")
    clean_folder(c, "output_data_dir", "*.h5")   # keras model weights
