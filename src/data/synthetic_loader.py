from pathlib import Path
import cv2

class SyntheticDataset:
    def __init__(self, root):
        self.paths = list(Path(root).glob("*.jpg"))

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        return img

    def __len__(self):
        return len(self.paths)
