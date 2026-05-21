from pickle import TUPLE
from typing import Callable, Optional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10:
    """
    实现功能：
    1.下载数据集包括训练集、测试集(自行选择是否联网下载) (√)
    2.对数据集进行预处理(如归一化、数据增强等) (√)
    3.将数据集打包为DataLoader形式返回 (√)
    调用实例：
    from load_data import CIFAR10

    loader = CIFAR10(
        dataset_path="./data",      
        transform=None,
        download_from_web=True,     
    )

    loader.process_dataset()
    train_loader, test_loader = loader.load_dataset(batch_size=64, shuffle=True)

    # 灰度：构造时传入 transform
    gray_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    loader_gray = CIFAR10(dataset_path="./data", transform=gray_transform)
    loader_gray.process_dataset()
    """

    def __init__(self, dataset_path: str = './data', transform: Optional[Callable] = None, download_from_web: bool = True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.download_from_web = download_from_web
        self.train_dataset = None
        self.test_dataset = None
        self.LOG_PREFIX = "CIFAR10"
        if transform is None:
            self._log("未提供 transform,使用默认 ToTensor + Normalize", level="WARN")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.download_dataset()  # 创建对象时自动完成下载与预处理

    def download_dataset(self) -> None:
        if self.download_from_web:
            self.train_dataset = datasets.CIFAR10(
                root=self.dataset_path, train=True, download=True)
            self.test_dataset = datasets.CIFAR10(
                root=self.dataset_path, train=False, download=True)
            self._log("联网下载成功", level="INFO")
        else:
            self._log("无法进行联网下载,将使用本地数据", level="WARN")
            self.train_dataset = datasets.CIFAR10(
                root=self.dataset_path, train=True, download=False)
            self.test_dataset = datasets.CIFAR10(
                root=self.dataset_path, train=False, download=False)
            self._check_dataset()

    def process_dataset(self) -> None:
        self._check_dataset()
        self.train_dataset.transform = self.transform
        self.test_dataset.transform = self.transform

    def load_dataset(self, batch_size: int = 128, shuffle: bool = True) -> tuple[DataLoader, DataLoader]:
        self._check_dataset()
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)
        self._log(
            f"数据集加载成功,训练集包括{len(train_loader.dataset)}个样本,测试集包括{len(test_loader.dataset)}个样本是,单张图像形状为{tuple(train_loader.dataset[0][0].shape)}", level="INFO")
        return train_loader, test_loader

    def _check_dataset(self) -> None:
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("未找到数据集,请检查是否下载或路径是否正确")

    def _log(self, context: str, level: str = "INFO") -> None:
        print(f"[{self.LOG_PREFIX}][{level}]: {context}")


if __name__ == "__main__":
    GRAY_TRANSFORM = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    def run(loader, ch: int):
        loader.process_dataset()
        tr, _ = loader.load_dataset(batch_size=64)
        x, y = next(iter(tr))
        assert len(loader.train_dataset) == 50_000 and len(
            loader.test_dataset) == 10_000
        assert loader.train_dataset.transform is loader.test_dataset.transform
        assert x.shape == (64, ch, 32, 32) and y.shape == (64,)
        print(f"通过 | batch={tuple(x.shape)}")

    print("=== CIFAR10 自检 ===")
    run(CIFAR10("./data", transform=None, download_from_web=True), ch=3)
    run(CIFAR10("./data", transform=GRAY_TRANSFORM, download_from_web=False), ch=1)

    print("-" * 30)
    try:
        local = CIFAR10("./data", download_from_web=False)
        local.process_dataset()
        local.load_dataset(64)
        print("通过 | 本地模式")
    except ValueError as e:
        print(f"跳过 | 本地模式 ({e})")
    print("=== 完成 ===")
