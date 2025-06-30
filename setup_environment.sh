#!/bin/bash

# Setup script for Inverse Game Discount Factor project
# This script creates a conda environment and sets up the Python-Julia integration

echo "Setting up Inverse Game Discount Factor environment..."

# Check if environment already exists
if conda env list | grep -q "inverse-game-discount-factor"; then
    echo "Environment already exists. Activating and installing kernel..."
    source activate inverse-game-discount-factor
else
    # Create conda environment from yml file
    echo "Creating conda environment..."
    conda env create -f conda-env.yml

    # Activate the environment
    echo "Activating environment..."
    source activate inverse-game-discount-factor

    # Install additional requirements if needed
    echo "Installing additional Python packages..."
    pip install -r requirements.txt

    # Setup PyJulia
    echo "Setting up PyJulia..."
    python -c "import julia; julia.install()"
fi

# Install IPython kernel for Cursor
echo "Installing IPython kernel..."
python -m ipykernel install --user --name=inverse-game-discount-factor --display-name="Python (inverse-game-discount-factor)"

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate inverse-game-discount-factor"
echo ""
echo "The Python kernel should now be available in Cursor!"
echo "You may need to restart Cursor to see the new kernel."