import os
from typing import Dict, List, Optional
import numpy as np
import swanlab
import torch
import torchvision.utils as vutils


class SwanLabLogger:
    """
    SwanLab 实验记录：构造 → init → create_experiment → log → finish。
    实现功能：
    1. 初始化 SwanLab 记录器 (√)                    SwanLabLogger() → init(project=..., mode=..., api_key=...)
    2. 创建实验 (√)                                 SwanLabLogger.create_experiment(experiment_name=..., config=..., tags=...)
    3. 记录训练/验证/测试指标 (√)                    SwanLabLogger.log_metrics({"loss": ..., "accuracy": ...}, step=...)
    4. 记录单张或多张图片 (√)                        SwanLabLogger.log_image(key=..., image=..., step=..., caption=...)
    5. 将常见数据类型转换为 SwanLab 可记录格式 (√)    SwanLabLogger._standardize_metrics(metrics=...)
    6. 结束实验，上传数据 (√)                        SwanLabLogger.finish()

    使用示例：
    logger = SwanLabLogger()
    logger.init(project_name="my_project", mode="cloud", api_key="your_api_key")
    logger.create_experiment(experiment_name="exp1", config={"lr": 0.001}, tags=["baseline"])
    for epoch in range(num_epochs):
        # 训练代码...
        logger.log_metrics({"loss": loss_value, "accuracy": acc_value}, step=epoch)
        if epoch % 10 == 0:
            img_grid = logger.make_cls_grid(batch_images)
            logger.log_image("train_images", img_grid, step=epoch, caption=f"Epoch {epoch} Images")
    logger.finish()
    """

    def __init__(self):
        """空构造，不连云端。参数在 init() 中设置。"""
        self.mode = "disabled"
        self.logdir = "./swanlog"
        self._started = False  # 是否已 create_experiment
        self.LOG_PREFIX = "SwanLabLogger"

    def init(
        self,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None,
        mode: str = "cloud",
        local_logdir: str = "./swanlog",
    ):
        self.project_name = project_name  # 项目名称，必填
        self.mode = mode
        self.logdir = local_logdir  # 好像在终端环境中配置过api后不用手动输出默认从环境变量里获取
        self.api_key = api_key or os.getenv("SWANLAB_API_KEY")
        self._started = False
        self._validate_params(self.mode, self.project_name,
                              self.api_key, self.logdir)  # 校验参数
   
        if self.mode == "cloud":
            swanlab.login(api_key=self.api_key)  # 云端模式需要登录
            self._log("已登录 SwanLab 云端", level="INFO")
        elif self.mode == "offline":
            self._log(f"离线模式，日志将保存在 '{self.logdir}'", level="INFO")
        else:
            self._log("已设置为 'disabled' 模式，无法上传数据。", level="WARNING")

    def create_experiment(
        self,
        experiment_name: str,  # 实验名称，必填
        config: Optional[Dict] = None,  # 实验配置（如超参数等），可选
        tags: Optional[List[str]] = None,  # 实验标签（如 "baseline", "resnet" 等），可选
    ):
        self.experiment_name = experiment_name  # 设置实验名称

        """创建一次实验（必填 experiment_name）。"""
        if self.project_name is None:
            raise RuntimeError("请先调用 init(project=...)")

        if self.mode == "disabled":
            self._log("当前模式为 'disabled'，无法创建实验。", level="WARNING")
            return

        swanlab.init(
            project=self.project_name,
            experiment_name=self.experiment_name,
            config=config or {},
            mode=self.mode,
            logdir=self.logdir,
            tags=tags,
        )
        self._started = True  # 标记已创建实验
        self._log(f"项目名称 '{self.project_name}', 实验 '{self.experiment_name}' 创建成功", level="INFO")

    def log_metrics(self, metrics: Dict[str, float], step: int, print_log: bool = True):
        """
        使用示例：
        logger.log_metrics(
            {"loss": loss_value, "accuracy": acc_value},
            step=epoch,
            print_log=True,
        )   
        """
        if not self._check_for_upload():
            return

        standardized_metrics = self._standardize_metrics(metrics)  # 标准化指标格式
        swanlab.log(standardized_metrics, step=step)  # 上传指标到 SwanLab
        if print_log:
            formatted_metrics = self._format_metrics_for_log(standardized_metrics)  # 格式化指标以便打印
            self._log_with_information(
                f"已记录指标: {formatted_metrics} (step={step})", project=self.project_name, experiment=self.experiment_name, level="INFO")

    def log_image(self, key: str, image: np.ndarray, step: int, caption: str = "", print_log: bool = True):
        """
        使用示例：
        logger.log_image(
            key="train/images",
            image=image,
            step=epoch,
            caption=f"Epoch {epoch}",
            print_log=True,
        )
        """
        if not self._check_for_upload():
            return
        swanlab.log({key: swanlab.Image(image, caption=caption)}, step=step)
        if print_log:
            self._log_with_information(f"已记录图像: {key} (step={step})", project=self.project_name, experiment=self.experiment_name, level="INFO")

    def log_images(self, key: str, images: list, step: int, captions: Optional[List[str]] = None, print_log: bool = True):
        """
        使用示例：
        logger.log_images(
            key="train/images",
            images=[img1, img2, img3],
            step=epoch,
            captions=[f"Image {i}" for i in range(3)],
            print_log=True,
        )
        """
        if not self._check_for_upload():
            return
        if captions is not None and len(captions) != len(images):
            raise ValueError("captions 的长度必须与 images 的长度一致")
        caps = captions or [""] * len(images)
        swanlab.log({key: [swanlab.Image(img, caption=c)
                    for img, c in zip(images, caps)]}, step=step)
        if print_log:
            self._log_with_information(f"已记录 {len(images)} 张图像: {key} (step={step})", project=self.project_name, experiment=self.experiment_name, level="DEBUG")

    def finish(self):
        if self._started:
            swanlab.finish()  # 结束实验，上传数据
            self._log_with_information("实验结束，数据已上传。", project=self.project_name, experiment=self.experiment_name, level="INFO")
        else:
            self._log("没有正在进行的实验，无需结束。", level="WARNING")
            return
        self._started = False

    @staticmethod
    def make_cls_grid(images: torch.Tensor, max_n: int = 8, normalize: bool = True) -> np.ndarray:
        """
        作用：将一批图像张量转换为可记录的图像格式（如 PNG）。适用于分类任务的图像展示。
        输入：images - 形状为 (B, C, H, W) 的图像张量，max_n - 最多展示的图像数量，normalize - 是否归一化图像像素值到 [0, 1]。
        输出：一个 NumPy 数组，表示拼接后的图像网格，适合直接记录到 SwanLab 中。

        使用示例：
        img_grid = logger.make_cls_grid(batch_images, max_n=16, normalize=True)
        logger.log_image("train/images", img_grid, step=epoch, caption=f"Epoch {epoch}"
        """
        images = images[:max_n].detach().cpu()
        grid = vutils.make_grid(images, nrow=max_n, normalize=normalize)
        grid = grid.permute(1, 2, 0).numpy()
        if grid.max() <= 1.0:
            grid = (grid * 255).clip(0, 255)
        return grid.astype(np.uint8)

    @staticmethod
    def mask_to_rgb(mask: np.ndarray, num_classes: int) -> np.ndarray:
        """
        作用：将语义分割的标签掩码转换为 RGB 图像，便于可视化。每个类别对应一种颜色。
        输入：mask - 形状为 (H, W) 的整数数组，表示每个像素的类别标签；num_classes - 类别总数，用于生成颜色映射。
        输出：一个形状为 (H, W, 3) 的 RGB 图像数组，每个像素的颜色根据其类别标签映射到一个 RGB 颜色。

        使用示例：
        label_rgb = SwanLabLogger.mask_to_rgb(label_mask, num_classes=20)
        logger.log_image("train/labels", label_rgb, step=epoch, caption=f"Epoch {epoch}")
        """
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", num_classes)
        return (cmap(mask % num_classes)[..., :3] * 255).astype(np.uint8)

    @staticmethod
    def make_seg_triplet(image: np.ndarray, label_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int) -> np.ndarray:
        """
        作用：将原始图像、标签掩码和预测掩码拼接成一个三联图，便于比较和可视化。标签和预测掩码会被转换为 RGB 图像。
        输入：image - 形状为 (H, W, C) 的原始图像；label_mask - 形状为 (H, W) 的标签掩码；pred_mask - 形状为 (H, W) 的预测掩码；num_classes - 类别总数，用于生成颜色映射。
        输出：一个形状为 (H, W*3, C) 的 RGB 图像数组，左侧是原始图像，中间是标签掩码的 RGB 可视化，右侧是预测掩码的 RGB 可视化。

        使用示例：
        triplet_image = SwanLabLogger.make_seg_triplet(image, label_mask, pred_mask, num_classes=20)
        logger.log_image("train/seg_triplet", triplet_image, step=epoch, caption=f"Epoch {epoch} Segmentation Triplet")
        """
        label_rgb = SwanLabLogger.mask_to_rgb(label_mask, num_classes)
        pred_rgb = SwanLabLogger.mask_to_rgb(pred_mask, num_classes)
        return np.concatenate([image, label_rgb, pred_rgb], axis=1)

    def _validate_params(self, mode: str, project: Optional[str], api_key: Optional[str], local_logdir: str) -> None:
        # 校验参数
        if mode not in {"cloud", "offline", "disabled"}:
            raise ValueError(
                f"mode 必须是 'cloud', 'offline' 或 'disabled'，但 got '{mode}'")
        if mode != "disabled" and not project:
            raise ValueError("mode 不为 'disabled' 时，project 不能为空")
        if mode == "cloud" and not (api_key or os.getenv("SWANLAB_API_KEY")):
            raise ValueError(
                "mode 为 'cloud' 时，必须提供 API Key。可通过参数 api_key 或环境变量 SWANLAB_API_KEY 设置。")
        if mode == "offline" and not os.path.isdir(local_logdir):
            os.makedirs(local_logdir, exist_ok=True)

    def _check_for_upload(self):
        if self.mode == "disabled":
            self._log("当前模式为 'disabled'，无法上传数据。", level="WARNING")
            return False
        if not self._started:
            raise RuntimeError("请先调用 create_experiment(...)")
        return True

    def _standardize_metrics(self, metrics: Dict) -> Dict[str, float]:
        standardized_metrics = {}
        for key, value in metrics.items():  # 只针对字典中的值进行处理，键保持不变
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.detach().cpu().item()
                else:
                    raise ValueError(f"指标 {key} 是多元素 Tensor,无法转换为单个标量")
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    raise ValueError(f"指标 {key} 是多元素 ndarray,无法转换为单个标量")
            elif isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, (int, float)):
                standardized_metrics[key] = value
            else:
                raise TypeError(f"指标 {key} 的类型不支持: {type(value)}")
        return standardized_metrics

    def _format_metrics_for_log(self, metrics: Dict[str, float], digits: int = 4) -> Dict[str, float]:
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics[key] = round(value, digits)  # 保留小数点后 digits 位
            else:
                formatted_metrics[key] = value

        return formatted_metrics

    def _log(self, context: str, level: str = "INFO") -> None:
        print(f"[{self.LOG_PREFIX}][{level}]: {context}")

    def _log_with_information(self, context: str, project: str, experiment: str, level: str = "INFO") -> None:
        print(f"[{self.LOG_PREFIX}][{project}][{experiment}][{level}]: {context}")
