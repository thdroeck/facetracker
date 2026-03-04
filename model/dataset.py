# ... (Imports and Dataset class remain at the top)

from torchvision import transforms
from dataloader import FaceDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # 1. Setup transforms
    data_transforms = transforms.Compose(
        [
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ]
    )

    # 2. Initialize
    my_dataset = FaceDataset(data_dir="./data", transform=data_transforms)

    # 3. Create the loader
    # Pro-tip: For debugging this error, set num_workers=0 first.
    # If it works with 0, then the 'main' guard below is definitely the fix.
    train_loader = DataLoader(
        my_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,  # Now this won't crash
        collate_fn=collate_fn,
    )

    # 4. Test it
    try:
        images, labels = next(iter(train_loader))
        print(f"Successfully loaded batch! Image shape: {images[0].shape}")
    except Exception as e:
        print(f"An error occurred: {e}")
