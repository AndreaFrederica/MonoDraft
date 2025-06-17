#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import argparse


def denoise_with_closing(
    input_path: str,
    output_path: str,
    threshold: int,
    fg_color: tuple[int, int, int],
    kernel_size: int = 5
) -> None:
    """
    1) 读入 RGB 图并计算亮度
    2) 亮度 ≤ threshold 视为前景
    3) 对前景掩码做闭运算填孔
    4) 前景染成 fg_color 并置 alpha=255，背景 alpha=0
    """
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)

    fg_mask = lum <= threshold

    mask_u8 = fg_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    fg2 = closed.astype(bool)

    h, w = lum.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3][fg2] = fg_color
    out[..., 3][fg2] = 255

    Image.fromarray(out, mode="RGBA").save(output_path)
    print(f"[+] Saved denoised image with closing: {output_path}")


def compute_stats(input_path: str, bins: int):
    """
    计算亮度统计、ASCII 直方图，并返回 Otsu 阈值
    """
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(int)

    hist, bin_edges = np.histogram(lum, bins=bins, range=(0, 255))
    max_count = hist.max() if hist.max() > 0 else 1
    print(f"亮度直方图 (bins={bins}):")
    for i in range(len(hist)):
        start = int(bin_edges[i])
        end = int(bin_edges[i+1] - 1)
        bar_len = int(hist[i] / max_count * 50)
        bar = '#' * bar_len if hist[i] > 0 else '-'
        print(f"{start:3d}-{end:3d}: {bar:<50} ({hist[i]})")

    min_l, max_l = lum.min(), lum.max()
    print(f"亮度范围: {min_l} ~ {max_l}")

    gray = lum.astype(np.uint8)
    otsu_thr, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    otsu_thr = int(otsu_thr)
    print(f"Otsu 推荐阈值: {otsu_thr}")

    return otsu_thr


def parse_color(s: str) -> tuple[int, int, int]:
    """解析 R,G,B 三元组字符串"""
    parts = s.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("颜色格式应为 R,G,B")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("颜色必须为整数")
    if any(c < 0 or c > 255 for c in rgb):
        raise argparse.ArgumentTypeError("颜色值应在 0-255 之间")
    return rgb


def main():
    parser = argparse.ArgumentParser(
        description="对彩色扫描件去底、降噪并填充小孔洞，输出 RGBA PNG"
    )
    parser.add_argument('input', help='输入图片路径（支持常见格式）')
    parser.add_argument('output', nargs='?', help='输出 PNG 路径，若使用 --stats 和 --auto 可省略')
    parser.add_argument('threshold', type=int, nargs='?', help='亮度阈值（0-255）')
    parser.add_argument('--fg', type=parse_color, default=(0, 0, 0), help='前景颜色 R,G,B，默认黑色')
    parser.add_argument('--kernel', type=int, default=5, help='闭运算结构元尺寸，默认 5')
    parser.add_argument('--stats', action='store_true', help='输出亮度统计及推荐阈值，然后退出')
    parser.add_argument('--bins', type=int, default=32, help='ASCII 直方图分区数量，默认 32')
    parser.add_argument('--auto', action='store_true', help='自动分析阈值、打印统计并执行降噪')

    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].lower() == 'help'):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.stats and not args.auto:
        compute_stats(args.input, args.bins)
        sys.exit(0)

    if args.auto:
        # 自动模式：先统计再降噪
        otsu_thr = compute_stats(args.input, args.bins)
        out = args.output or f"{Path(args.input).stem}_auto.png"
        print(f"Auto 模式：使用阈值 {otsu_thr} 进行降噪，输出 {out}")
        denoise_with_closing(args.input, out, otsu_thr, args.fg, args.kernel)
        sys.exit(0)

    # 普通降噪模式
    if not args.output or args.threshold is None:
        parser.error("降噪模式需要 <output> 和 <threshold> 参数，或使用 --stats/--auto 模式")

    if args.threshold < 0 or args.threshold > 255:
        parser.error("阈值应在 0-255 之间")

    denoise_with_closing(args.input, args.output, args.threshold, args.fg, args.kernel)

if __name__ == "__main__":
    main()
