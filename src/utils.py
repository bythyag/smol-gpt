import torch
import random
import numpy as np
import os

#------------------------------------
# Utility Functions
#------------------------------------

def set_seed(seed):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mount_drive(mount_path='/content/drive'):
    """ Mounts Google Drive in Colab environment. """
    try:
        from google.colab import drive
        if not os.path.exists(mount_path) or not os.listdir(mount_path):
             print(f"Mounting Google Drive at {mount_path}...")
             drive.mount(mount_path)
             print("Drive mounted successfully.")
        else:
             print("Drive already mounted.")
    except ImportError:
        print("Not in Colab environment, skipping Drive mount.")
    except Exception as e:
        print(f"Error mounting Drive: {e}")

def get_device():
    """ Gets the appropriate device (CUDA or CPU). """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        # You could add specific GPU selection logic here if needed
        return torch.device('cuda')
    else:
        print("CUDA not available. Using CPU.")
        return torch.device('cpu')