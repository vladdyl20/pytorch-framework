from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom PyTorch Dataset for image classification.

    Expected directory structure:

    root_dir/
        pizza/
            image1.jpg
            image2.jpg
        steak/
            image1.jpg
        sushi/
            image1.jpg
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        # Get class names from folder names
        self.classes = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        if not self.classes:
            raise RuntimeError(
                f"No class directories found in {self.root_dir}"
            )

        # Create class -> integer mapping
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        }

        self.samples = []

        # Collect all image paths and their labels
        for class_name in self.classes:
            class_directory = self.root_dir / class_name

            for image_path in class_directory.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in valid_extensions
                ):
                    label = self.class_to_idx[class_name]

                    self.samples.append(
                        (image_path, label)
                    )

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.root_dir}"
            )

    def __len__(self) -> int:
        """Return number of images in the dataset."""

        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Load one image and its corresponding class label.
        """

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label