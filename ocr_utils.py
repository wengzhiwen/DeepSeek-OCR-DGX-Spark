#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR 共享工具函数模块

包含图片处理、结果解析、边界框绘制和结果保存等通用功能。
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


def load_image(image_path):
    """
    加载图片并修正 EXIF 方向。

    手机拍摄的图片可能包含 EXIF 方向信息，需要根据该信息旋转图片
    以确保图片方向正确。

    Args:
        image_path: 图片文件路径

    Returns:
        PIL.Image: 方向修正后的图片，失败返回 None
    """
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"加载图片错误: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def parse_grounding_tags(text):
    """
    解析 OCR 结果中的 grounding 标签。

    DeepSeek-OCR 输出格式:
        <|ref|>类型<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>

    其中坐标是归一化到 [0, 999] 范围的整数。

    Args:
        text: OCR 输出的原始文本

    Returns:
        tuple: (所有匹配, 图片类型匹配, 其他类型匹配)
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """
    从 grounding 标签中提取坐标和标签类型。

    Args:
        ref_text: 正则匹配结果元组 (完整匹配, 标签类型, 坐标字符串)
        image_width: 原始图片宽度
        image_height: 原始图片高度

    Returns:
        tuple: (标签类型, 坐标列表) 或 None
    """
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"解析坐标错误: {e}")
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, output_path):
    """
    在图片上绘制检测到的边界框。

    根据不同的元素类型（title, text, image 等）使用不同颜色和线宽绘制边界框，
    并在框上方标注类型名称。同时提取 image 类型的区域保存为单独文件。

    Args:
        image: PIL.Image 原始图片
        refs: grounding 标签匹配列表
        output_path: 输出目录路径

    Returns:
        PIL.Image: 绘制了边界框的图片
    """
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    img_idx = 0
    images_dir = Path(output_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width,
                                                   image_height)
            if result:
                label_type, points_list = result
                color = (np.random.randint(0, 200), np.random.randint(0, 200),
                         np.random.randint(0, 255))
                color_a = color + (20, )

                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(str(images_dir / f"{img_idx}.jpg"))
                        except Exception as e:
                            print(f"保存裁剪图片错误: {e}")
                        img_idx += 1

                    try:
                        width = 4 if label_type == 'title' else 2
                        draw.rectangle([x1, y1, x2, y2],
                                       outline=color,
                                       width=width)
                        draw2.rectangle([x1, y1, x2, y2],
                                        fill=color_a,
                                        outline=(0, 0, 0, 0),
                                        width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0),
                                                  label_type,
                                                  font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([
                            text_x, text_y, text_x + text_width,
                            text_y + text_height
                        ],
                                       fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y),
                                  label_type,
                                  font=font,
                                  fill=color)
                    except:
                        pass
        except:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def save_ocr_results(final_output, image, image_name, output_path):
    """
    保存 OCR 结果到文件。

    保存的文件包括：
    - {image_name}_ori.mmd: 原始输出
    - {image_name}.mmd: 处理后的 markdown
    - {image_name}.md: Markdown 格式
    - {image_name}.txt: 文本格式
    - {image_name}_with_boxes.jpg: 带边界框的图片

    Args:
        final_output: OCR 输出的原始文本
        image: PIL.Image 原始图片
        image_name: 图片文件名（不含扩展名）
        output_path: 输出目录路径

    Returns:
        str: 处理后的 OCR 文本结果
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(parents=True, exist_ok=True)

    matches_ref, matches_images, matches_other = parse_grounding_tags(
        final_output)

    result_image = draw_bounding_boxes(image, matches_ref, str(output_path))

    processed_output = final_output
    for idx, match_image in enumerate(tqdm(matches_images, desc="处理图片引用")):
        processed_output = processed_output.replace(
            match_image, f'![](images/{idx}.jpg)\n')

    for idx, match_other in enumerate(tqdm(matches_other, desc="处理其他引用")):
        processed_output = processed_output.replace(match_other, '').replace(
            '\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    with open(output_path / f'{image_name}_ori.mmd', 'w',
              encoding='utf-8') as f:
        f.write(final_output)

    with open(output_path / f'{image_name}.mmd', 'w', encoding='utf-8') as f:
        f.write(processed_output)

    with open(output_path / f'{image_name}.md', 'w', encoding='utf-8') as f:
        f.write(processed_output)

    with open(output_path / f'{image_name}.txt', 'w', encoding='utf-8') as f:
        f.write(processed_output)

    result_image.save(str(output_path / f'{image_name}_with_boxes.jpg'))

    return processed_output
