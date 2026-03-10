import argparse

from torchvision import transforms

from model.architectures.net import Net
from model.architectures.vgg19 import VGG19
from model.datasets.facedataset import FaceDataset
from model.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a face landmark detection model"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training the model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="freak2209",
        help="Dataloader to use for training",
    )

    parser.add_argument(
        "--dataset", type=str, default="./data", help="Dataroot to use for training"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model architecture to use for training",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Path to save the trained model",
    )

    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=10,
        help="Save the model every n epochs",
    )

    args = parser.parse_args()

    train(args)
