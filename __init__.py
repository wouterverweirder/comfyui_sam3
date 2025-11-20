"""Top-level package for comfyui_sam3."""

import os
import subprocess
import sys

# Install SAM3 from local src/sam3 directory
sam3_dir = os.path.join(os.path.dirname(__file__), "src", "sam3")
if os.path.exists(sam3_dir):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", sam3_dir], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: Failed to install SAM3: {e}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    
]

__author__ = """Wouter Verweirder"""
__email__ = "wouter.verweirder@gmail.com"
__version__ = "0.0.1"

from .src.comfyui_sam3.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_sam3.nodes import NODE_DISPLAY_NAME_MAPPINGS


