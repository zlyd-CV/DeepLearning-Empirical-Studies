import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from load_data import CIFAR10
from model import AlexNet, ResNet152
from utils import SwanLabLogger


def run_epoch(model, loader, criterion, device, optimizer=None, desc="Train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=desc, leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = y.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
            total += n
            bar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def build_model(name, input_channels):
    if name == "alexnet":
        return AlexNet(num_classes=10, input_channels=input_channels)
    if name == "resnet152":
        return ResNet152(num_classes=10, input_channels=input_channels)
    raise ValueError(f"未知模型: {name}")


def build_transform(gray):
    if gray:
        return transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return None


def sample_image_grid(loader, max_n=4):
    images, _ = next(iter(loader))
    return SwanLabLogger.make_cls_grid(images, max_n=max_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "resnet152"], default="resnet152")
    parser.add_argument("--gray", action="store_true", help="是否使用灰度图像", default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--download", action="store_true", help="是否从网络下载数据集", default=False)
    parser.add_argument("--swan-mode", choices=["cloud", "offline", "disabled"], default="cloud")
    parser.add_argument("--project", default="Github_Exp01")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 1 if args.gray else 3
    experiment_name = f"{args.model}-{'gray' if args.gray else 'rgb'}"

    data = CIFAR10(args.data, transform=build_transform(args.gray), download_from_web=args.download)
    data.process_dataset()
    train_loader, test_loader = data.load_dataset(batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = SwanLabLogger()
    logger.init(project_name=args.project, mode=args.swan_mode,api_key="zaCD5rdZT8meNAXhVVM0h")
    logger.create_experiment(experiment_name, config=vars(args) | {"device": str(device)})

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer, f"Train {epoch}/{args.epochs}"
            )
            test_loss, test_acc = run_epoch(
                model, test_loader, criterion, device, None, f"Test  {epoch}/{args.epochs}"
            )

            logger.log_metrics({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            logger.log_image(
                key="test/images",
                image=sample_image_grid(test_loader, max_n=4),
                step=epoch,
                caption=f"Epoch {epoch}",
            )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
