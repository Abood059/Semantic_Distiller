import sys
import subprocess
import os
from pathlib import Path


def main():
    """Bootstrap script for Semantic Distiller."""
    project_root = Path(__file__).parent
    venv_dir = project_root / ".venv"
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    
    # Get the python executable in the venv
    if os.name == 'nt':  # Windows
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix-like
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    
    # Run main.py with all arguments passed through
    print("Starting Semantic Distiller...")
    subprocess.run([str(python_exe), "main.py"] + sys.argv[1:], check=True)


if __name__ == "__main__":
    main()
