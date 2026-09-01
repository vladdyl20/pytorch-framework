from pathlib import Path
import zipfile

import requests


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "mrdbourke/pytorch-deep-learning/main/data/pizza_steak_sushi.zip"
)


def download_dataset(
    url: str = DATA_URL,
    target_dir: str = "data",
    dataset_name: str = "pizza_steak_sushi",
) -> Path:
    """Download and extract the dataset."""

    target_path = Path(target_dir)
    dataset_path = target_path / dataset_name
    zip_path = target_path / f"{dataset_name}.zip"

    if dataset_path.exists():
        print(f"[INFO] Dataset already exists: {dataset_path}")
        return dataset_path

    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading dataset from:\n{url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    zip_path.write_bytes(response.content)

    print("[INFO] Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_path)

    zip_path.unlink()

    print(f"[INFO] Dataset ready: {dataset_path}")

    return dataset_path


if __name__ == "__main__":
    download_dataset()