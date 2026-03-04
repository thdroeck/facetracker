from torchvision import transforms
import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from .dataloader import FaceDataset
from .simple import Net


def collate_fn(batch):
    images, labels = zip(*batch)
    # Stack images into a single tensor, but keep labels as a tuple
    return torch.stack(images), labels


def train():

    data_transforms = transforms.Compose(
        [
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ]
    )

    face_dataset = FaceDataset(data_dir="./data", transform=data_transforms)

    train_loader = DataLoader(
        face_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,  # Now this won't crash
        collate_fn=collate_fn,
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        running_loss = 0.0

        # Wrap the train_loader with tqdm
        # desc: adds a prefix text to the bar
        # unit: labels each step as a 'batch'
        loop = tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"
        )

        for i, (inputs, labels) in loop:
            # 1. Prepare targets
            # If your collate_fn returns a tuple of tensors, stack them:
            targets = torch.stack(labels).float()
            if targets.shape[1] == 5:
                targets = targets[:, 1:]

            # 2. Zero the parameter gradients
            optimizer.zero_grad()

            # 3. Forward + Backward + Optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 4. Update Progress Bar
            running_loss += loss.item()

            # This updates the right side of the progress bar with the current average loss
            loop.set_postfix(loss=running_loss / (i + 1))

        print(
            f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(train_loader):.4f}"
        )

    print("Finished Training")

    # save model
    torch.save(net.state_dict(), "face_landmark_model.pth")


if __name__ == "__main__":
    train()
