import os
import urllib.request
import sys

def download_sam_model(model_type="vit_b"):
    """
    Download the SAM model checkpoint if it doesn't exist.
    
    Args:
        model_type: The SAM model type to download (default: "vit_b")
    
    Returns:
        Path to the downloaded model checkpoint
    """
    model_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    if model_type not in model_urls:
        raise ValueError(f"Model type {model_type} not supported. Choose from: {list(model_urls.keys())}")
    
    url = model_urls[model_type]
    filename = os.path.basename(url)
    
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    else:
        print(f"Model file {filename} already exists")
        
    return filename

def ensure_segment_anything():
    """
    Ensure that segment_anything module is available, installing if needed
    """
    try:
        import segment_anything
        print("segment_anything module is already installed")
    except ImportError:
        print("segment_anything module not found, installing from local directory...")
        try:
            # Try to install from local directory if available
            if os.path.exists("segment-anything"):
                print("Installing segment-anything from local directory")
                os.system(f"{sys.executable} -m pip install -e segment-anything")
            else:
                print("Installing segment-anything from GitHub")
                os.system(f"{sys.executable} -m pip install git+https://github.com/facebookresearch/segment-anything.git")
            
            # Verify installation
            import segment_anything
            print("segment_anything module successfully installed")
        except Exception as e:
            print(f"Error installing segment_anything: {e}")
            return False
    return True

if __name__ == "__main__":
    # Ensure segment-anything is installed
    if ensure_segment_anything():
        # Download the default model
        download_sam_model()