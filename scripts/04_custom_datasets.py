from pytorch_framework.data.datasets import CustomImageDataset
from pytorch_framework.data.transforms import get_basic_transform
from pytorch_framework.data.loaders import create_dataloaders


def main():
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    transform = get_basic_transform(
        image_size=64
    )

    dataset = CustomImageDataset(
        root_dir=train_dir,
        transform=transform,
    )

    image, label = dataset[0]

    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Class name: {dataset.classes[label]}")

    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=transform,
        test_transform=transform,
        batch_size=32,
        num_workers=0,
    )

    images, labels = next(
        iter(train_loader)
    )

    print()
    print(f"Classes from DataLoader: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")


if __name__ == "__main__":
    main()