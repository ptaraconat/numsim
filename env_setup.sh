#!/bin/bash
sudo apt-get update 
sudo apt-get install python3-gmsh
python -m venv devenv
source devenv/bin/activate
python -m pip install -r requirements.txt