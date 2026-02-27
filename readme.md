# Surveillance Robot

## Features
- Extract Faces Embeddings
- Build a Recognition Engine

## Installation

### Prerequisities
- Python 3.x
- pip

### Setup
```bash
    # Clone the repository 
    git clone https://github.com/AhmadYamen/Surveillance-Robot.git

    # Navigate to the project directory
    cd surveillance-robot

    # Create virtual environment
    python -m venv venv
    venv/Scripts/activate (on Windows)

    # Install dependencies
    pip install -r requirements.txt
```
## Usage
```bash
   # Extract Images Embeddings by running the following script
   # After labelling the extracted embeddings, export them
   python training_app.py

   # Write the following line to run the Recognition Engine
   # Notice you have to change the CAM_INDEX in config.yaml to (0 or 1) according to your attached camera
   python reco_engine.py
   