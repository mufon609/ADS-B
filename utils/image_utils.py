import os

def to_img_url(fs_path: str) -> str:
    """Converts a filesystem image path to its corresponding dashboard URL."""
    return "/images/" + os.path.basename(str(fs_path)) if fs_path else ""
