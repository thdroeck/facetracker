from torchvision import transforms
import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch

from model.architectures.net import Net
from model.architectures.vgg19 import VGG19
from model.datasets.facedataset import FaceDataset


def getDataloader(args):
    if args.dataloader == "freak2209":
        data_transforms = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
            ]
        )
        return FaceDataset(data_dir=args.dataset, transform=data_transforms)
    else:
        raise ValueError(f"Unsupported dataloader: {args.dataloader}")


def getModel(args):
    if args.model == "net":
        return Net()
    elif args.model == "vgg19":
        return VGG19()
    else:
        raise ValueError(f"Unsupported model architecture: {args.model}")


def collate_fn(batch):
    images, labels = zip(*batch)
    # Stack images into a single tensor, but keep labels as a tuple
    return torch.stack(images), labels


def train(args):
    torch.mps.empty_cache()

    model_name = f"{args.model}_{args.dataloader}_{args.epochs}epochs_{args.batch_size}batch_{args.learning_rate}lr"

    face_dataset = getDataloader(args)

    # 2. Calculate lengths (1/3 for test, 2/3 for train)
    test_size = len(face_dataset) // 3
    train_size = len(face_dataset) - test_size

    # 3. Perform the random split
    train_dataset, test_dataset = random_split(
        face_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(
            42
        ),  # Optional: ensures the split is the same every time
    )

    # 4. Create separate DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = getModel(args)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device=device).half()  # Convert model to float16 and move to device
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    for epoch in range(args.epochs):
        running_loss = 0.0

        # Wrap the train_loader with tqdm
        # desc: adds a prefix text to the bar
        # unit: labels each step as a 'batch'
        loop = tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"
        )

        for i, (inputs, labels) in loop:
            inputs = inputs.to(
                device=device
            ).half()  # Convert inputs to float16 and move to device
            # 1. Prepare targets
            # If your collate_fn returns a tuple of tensors, stack them:
            targets = (
                torch.stack(labels).float().to(device=device).half()
            )  # Convert targets to float16 and move to device
            if targets.shape[1] == 5:
                targets = targets[:, 1:]

            # 2. Zero the parameter gradients
            optimizer.zero_grad()

            # 3. Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 4. Update Progress Bar
            running_loss += loss.item()

            # This updates the right side of the progress bar with the current average loss
            loop.set_postfix(loss=running_loss / (i + 1))

        if (epoch + 1) % args.save_every_n_epochs == 0:
            cummulative_test_loss = 0.0
            for test_inputs, test_labels in test_loader:
                test_targets = torch.stack(test_labels).float()
                if test_targets.shape[1] == 5:
                    test_targets = test_targets[:, 1:]

                with torch.no_grad():
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_targets)
                    cummulative_test_loss += test_loss.item()

            print(
                f"Epoch {epoch + 1} completed. Average Test Loss: {cummulative_test_loss / len(test_loader):.4f}"
            )

            torch.save(
                model.state_dict(),
                f"{args.save_path}/{model_name}_{epoch + 1}.pth",
            )

        print(
            f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(train_loader):.4f}"
        )

    print("Finished Training")

    # save model
    torch.save(model.state_dict(), f"{args.save_path}/{model_name}.pth")
