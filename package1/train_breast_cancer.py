"""Train a U-Net to segment lesions in the BUSI breast ultrasound dataset.

This educational example is not a medical diagnostic tool.

学习路线：main 读取参数 -> train 准备数据 -> run_epoch 训练 -> 测试并保存。
本程序学习的是“给病灶涂色”（分割），不是判断整张图片是否患癌。
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BUSIDataset(Dataset):
    """每次取出一张超声图及其标准答案（同名病灶掩膜）。"""
    def __init__(self, samples, image_size, augment=False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("L")
        with Image.open(mask_path) as source:
            mask = source.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image and mask sizes differ for {image_path.name}: "
                f"{image.size} != {mask.size}"
            )

        # 数据增强：只在训练时随机左右翻转；图片与答案必须一起翻转。
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        # 图像用平滑插值；掩膜用最近邻，避免插值产生假的标签值。
        image = resize_with_padding(
            image, self.image_size, Image.Resampling.BILINEAR
        )
        mask = resize_with_padding(
            mask, self.image_size, Image.Resampling.NEAREST
        )

        # 灰度从 0~255 缩放到 0~1；掩膜中非零像素是病灶，零是背景。
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        # PyTorch 要求单张灰度图的形状为 [通道数=1, 高, 宽]。
        return (
            torch.from_numpy(image_array).unsqueeze(0),
            torch.from_numpy(mask_array).unsqueeze(0),
        )


def resize_with_padding(image, size, resample):
    """等比例缩放并补黑边，得到 size × size，而不是把图像拉伸变形。"""
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    image = image.resize(new_size, resample=resample)
    left = (size - new_size[0]) // 2
    top = (size - new_size[1]) // 2
    return ImageOps.expand(
        image,
        border=(left, top, size - new_size[0] - left, size - new_size[1] - top),
        fill=0,
    )


def find_samples(data_dir):
    # labels 里的文件是像素级答案，不是另一批用来训练的超声照片。
    image_dir = data_dir / "BUSI" / "images"
    mask_dir = data_dir / "BUSI" / "labels"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            "BUSI files not found. Expected "
            f"{image_dir} and {mask_dir}."
        )

    image_paths = sorted(image_dir.glob("*.png"))
    samples = []
    missing_masks = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if mask_path.is_file():
            samples.append((image_path, mask_path))
        else:
            missing_masks.append(image_path.name)

    if missing_masks:
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} segmentation masks, e.g. "
            f"{', '.join(missing_masks[:5])}"
        )
    if not samples:
        raise FileNotFoundError(f"No PNG images found in {image_dir}.")
    return samples


def stratified_split(samples, seed, val_fraction, test_fraction):
    """分别打乱良性、恶性图片，使各集合都包含这两种图像。"""
    # 这是按图片划分，并不是按患者划分；不能据此声称临床泛化能力。
    groups = {"benign": [], "malignant": []}
    for sample in samples:
        name = sample[0].name.lower()
        if name.startswith("benign_"):
            groups["benign"].append(sample)
        elif name.startswith("malignant_"):
            groups["malignant"].append(sample)
        else:
            raise ValueError(
                f"Cannot infer BUSI class from image filename: {sample[0].name}"
            )

    rng = random.Random(seed)
    split_samples = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        val_count = max(1, round(len(group) * val_fraction))
        test_count = max(1, round(len(group) * test_fraction))
        if val_count + test_count >= len(group):
            raise ValueError("Validation and test fractions leave no training data.")
        split_samples["val"].extend(group[:val_count])
        split_samples["test"].extend(group[val_count : val_count + test_count])
        split_samples["train"].extend(group[val_count + test_count :])

    for split in split_samples.values():
        rng.shuffle(split)
    return split_samples


class ConvBlock(nn.Module):
    """两个卷积层提取局部纹理，ReLU 增加模型表达复杂规律的能力。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet(nn.Module):
    """U-Net：先缩小图像提取特征，再放大并融合细节，预测病灶位置。"""
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(1, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.encoder4 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, inputs):
        # 编码器逐级缩小空间尺寸；保留中间结果供解码器恢复边界细节。
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        center = self.bottleneck(self.pool(encoder4))
        # 跳跃连接：把放大的特征和相同尺寸的编码器特征拼接在一起。
        decoded = self.decoder4(
            torch.cat((self.up4(center), encoder4), dim=1)
        )
        decoded = self.decoder3(
            torch.cat((self.up3(decoded), encoder3), dim=1)
        )
        decoded = self.decoder2(
            torch.cat((self.up2(decoded), encoder2), dim=1)
        )
        decoded = self.decoder1(
            torch.cat((self.up1(decoded), encoder1), dim=1)
        )
        # 输出 logits（原始分数），还不是概率；后面用 sigmoid 转为概率。
        return self.output(decoded)


def dice_loss(logits, targets):
    # Dice 衡量预测病灶与真实病灶的重叠；损失为 1-Dice，越小越好。
    # 加 1 用于平滑，避免分母为零；训练时使用连续概率以便求导。
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(
        dim=(1, 2, 3)
    )
    dice = (2 * intersection + 1) / (denominator + 1)
    return 1 - dice.mean()


def calculate_dice(model, loader, device):
    # 评估时把概率 >= 0.5 的像素看作病灶，并跨整个集合累计重叠。
    model.eval()
    intersection = 0.0
    denominator = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = (torch.sigmoid(model(images)) >= 0.5).float()
            intersection += (predictions * masks).sum().item()
            denominator += predictions.sum().item() + masks.sum().item()
    if denominator == 0:
        return 1.0
    return (2 * intersection) / denominator


def calculate_pixel_accuracy(model, loader, device):
    # 像素准确率 = 判对的像素数 / 全部像素数，包含背景及补黑边区域。
    # 背景很多时准确率可能虚高，因此必须结合 Dice 看。
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = (torch.sigmoid(model(images)) >= 0.5).float()
            correct += (predictions == masks).sum().item()
            total += masks.numel()
    if total == 0:
        raise ValueError("Cannot calculate accuracy on an empty dataset.")
    return correct / total


def run_epoch(model, loader, device, optimizer=None):
    """遍历一个集合一次；传入 optimizer 才会更新模型，否则仅评估。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    loss_function = nn.BCEWithLogitsLoss()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            # BCE 关注逐像素判断，Dice loss 关注病灶整体重叠，两者相加。
            loss = loss_function(logits, masks) + dice_loss(logits, masks)
            if training:
                # 清除上一批梯度 -> 计算本批梯度 -> 按学习率更新参数。
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(logits.detach()) >= 0.5).float()
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
    if total_pixels == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")
    return total_loss / len(loader.dataset), correct_pixels / total_pixels


def train(args):
    # 固定随机种子便于比较实验；不同设备仍可能出现数值差异。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    samples = find_samples(args.data_dir)
    splits = stratified_split(
        samples, args.seed, args.val_fraction, args.test_fraction
    )
    print(
        f"Images: {len(samples)} | train: {len(splits['train'])} | "
        f"validation: {len(splits['val'])} | test: {len(splits['test'])}"
    )

    # DataLoader 把单张图片组成 batch。只有训练集使用随机翻转和打乱。
    loaders = {}
    for split, split_samples in splits.items():
        dataset = BUSIDataset(
            split_samples,
            args.image_size,
            augment=(split == "train"),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

    # 安装了可用的 CUDA 版 PyTorch 且有兼容 GPU 时才会使用 GPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_dice = -1.0
    best_state = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "busi_training_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_pixel_accuracy",
                "val_loss",
                "val_pixel_accuracy",
                "val_dice",
            ),
        )
        writer.writeheader()
        # 每轮先学习训练集，再用验证集检查；此时不查看测试集结果。
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], device, optimizer
            )
            val_loss, val_accuracy = run_epoch(model, loaders["val"], device)
            val_dice = calculate_dice(model, loaders["val"], device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_pixel_accuracy": f"{train_accuracy:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_pixel_accuracy": f"{val_accuracy:.6f}",
                    "val_dice": f"{val_dice:.6f}",
                }
            )
            metrics_file.flush()
            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"train loss: {train_loss:.4f}, train pixel acc: "
                f"{train_accuracy:.4f}, val loss: {val_loss:.4f}, "
                f"val pixel acc: {val_accuracy:.4f}, val Dice: {val_dice:.4f}",
                flush=True,
            )
            if val_dice > best_val_dice:
                # 按验证 Dice 选最优模型，而不是选训练准确率最高的模型。
                # 此处只在内存留副本；文件在全部轮次结束后才保存。
                best_val_dice = val_dice
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    # 用最优验证模型在未参与参数学习的测试集上做最后评估。
    model.load_state_dict(best_state)
    test_dice = calculate_dice(model, loaders["test"], device)
    test_accuracy = calculate_pixel_accuracy(model, loaders["test"], device)
    checkpoint_path = args.output_dir / "busi_unet.pt"
    torch.save(
        {
            "state_dict": best_state,
            "architecture": "UNet",
            "dataset": "BUSI",
            "image_size": args.image_size,
            "mask_threshold": 0.5,
            "class_names": ("benign", "malignant"),
            "best_validation_dice": best_val_dice,
            "test_dice": test_dice,
            "test_pixel_accuracy": test_accuracy,
        },
        checkpoint_path,
    )
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Test pixel accuracy: {test_accuracy:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
    print(f"Epoch metrics saved to: {metrics_path.resolve()}")


def predict_image(args):
    """加载已训练权重，只预测一张图，不修改模型参数。"""
    import matplotlib.pyplot as plt

    checkpoint_path = args.weights or args.output_dir / "busi_unet.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {checkpoint_path}. "
            "Train first or supply --weights."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "UNet":
        raise ValueError("Expected a BUSI UNet checkpoint produced by this script.")
    size = checkpoint["image_size"]
    if not isinstance(size, int) or size < 32 or size % 16:
        raise ValueError("Checkpoint image_size must be a multiple of 16, at least 32.")
    threshold = checkpoint["mask_threshold"]
    if not 0 < threshold < 1:
        raise ValueError("Checkpoint mask_threshold must be between zero and one.")

    model = UNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(args.predict) as source:
        original = source.convert("L")
    prepared = resize_with_padding(original, size, Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32).copy() / 255.0
    inputs = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))[0, 0].cpu().numpy()

    # 去掉训练预处理的黑边，再还原到原图尺寸，避免叠加图位置错位。
    width, height = original.size
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    cropped = probabilities[top : top + new_height, left : left + new_width]
    restored = Image.fromarray(cropped).resize(
        original.size, resample=Image.Resampling.BILINEAR
    )
    prediction = np.asarray(restored) >= threshold
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    overlay[prediction] = (1.0, 0.0, 0.0, 0.45)

    figure, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original ultrasound")
    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted lesion mask")
    axes[2].imshow(original, cmap="gray")
    axes[2].imshow(overlay)
    axes[2].set_title("Prediction overlay (red)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{args.predict.name} | weights: {checkpoint_path.name}")
    figure.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{args.predict.stem}_prediction.png"
    mask_path = args.output_dir / f"{args.predict.stem}_mask.png"
    figure.savefig(figure_path, dpi=150)
    Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
    print(f"Weights loaded: {checkpoint_path.resolve()}")
    print(f"Prediction figure saved to: {figure_path.resolve()}")
    print(f"Predicted mask saved to: {mask_path.resolve()}")
    print("Segmentation demonstration only; not a cancer diagnosis.")
    if not args.no_show:
        plt.show()
    plt.close(figure)


def main():
    # 默认目录以脚本自身为基准，方便在 IDE 或其他工作目录启动此副本。
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a U-Net lesion segmenter on the local BUSI dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data" / "BUSI")
    parser.add_argument("--output-dir", type=Path, default=package_dir / "outputs" / "BUSI")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predict", type=Path,
        help="Predict one image using saved weights instead of training.",
    )
    parser.add_argument("--weights", type=Path, help="Path to a trained UNet .pt file.")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save prediction plots without opening a window (prediction only).",
    )
    args = parser.parse_args()

    if args.predict is not None:
        predict_image(args)
        return
    if args.weights is not None or args.no_show:
        parser.error("--weights and --no-show require --predict")

    # U-Net 下采样四次，每次尺寸减半，因此输入边长必须是 16 的倍数。
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("--epochs and --batch-size must be positive; --workers >= 0")
    if args.image_size < 32 or args.image_size % 16:
        parser.error("--image-size must be a multiple of 16 and at least 32")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        parser.error("Validation and test fractions must be greater than zero")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("Validation and test fractions must sum to less than one")

    train(args)


if __name__ == "__main__":
    # Windows 使用多进程加载数据时也需要这个入口保护。
    main()
