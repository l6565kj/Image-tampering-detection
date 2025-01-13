import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
from datetime import datetime


def ela_check(input_image_path, quality=95):
    """
    Perform Error Level Analysis (ELA) on an image and check if it is suspicious.
    Returns True if the image is suspicious, otherwise False.
    """
    original = Image.open(input_image_path)
    if original.mode != 'RGB':
        original = original.convert('RGB')

    # Save the image with specified quality to calculate compression differences
    temp_path = 'temp.jpg'
    original.save(temp_path, 'JPEG', quality=quality)

    # Reopen the recompressed image
    recompressed = Image.open(temp_path)

    # Calculate the difference between the original and recompressed images
    diff = ImageChops.difference(original, recompressed)

    # Analyze the difference levels
    diff_stat = diff.getextrema()  # Returns min and max for each channel
    max_diff = max([v[1] for v in diff_stat if v is not None])  # Get max difference
    os.remove(temp_path)  # Cleanup temporary file

    return max_diff > 50  # Threshold for "suspicious" detection


def annotate_image(input_image_path, output_image_path, scale=10, suspicious=False, direction='right'):
    """
    Annotate the original image with red lines and stitch the processed result to the original image.
    Text annotations automatically adjust to resolution and maintain a safe margin.
    """
    original = Image.open(input_image_path)
    if original.mode != 'RGB':
        original = original.convert('RGB')

    # Save the image as a temporary file for compression
    temp_path = 'temp.jpg'
    original.save(temp_path, 'JPEG', quality=95)

    # Reopen the recompressed image
    recompressed = Image.open(temp_path)

    # Calculate the difference between the original and recompressed images
    diff = ImageChops.difference(original, recompressed)

    # Enhance the difference
    enhancer = ImageEnhance.Brightness(diff)
    diff_enhanced = enhancer.enhance(scale)

    # Convert the enhanced image to OpenCV format
    diff_cv = cv2.cvtColor(np.array(diff_enhanced), cv2.COLOR_RGB2BGR)

    # Create a separate layer for red annotations
    annotation_layer = np.zeros_like(diff_cv, dtype=np.uint8)

    # Add metadata text dynamically sized based on resolution
    img_width, img_height = original.size
    font_scale = max(0.5, min(img_width / 1000, img_height / 1000))  # Scale font based on resolution
    text_thickness = max(1, int(font_scale * 2))
    line_spacing = int(font_scale * 30)  # Dynamic line spacing
    safe_margin = max(20, int(font_scale * 20))  # Safe margin based on resolution

    metadata = [
        "Digital Evidence Analysis",
        "Powered by 6565TECH",
        "Suspicious regions detected" if suspicious else "No suspicious regions found"
    ]

    # Ensure text starts within screen boundaries with safe margin
    x_start = safe_margin
    y_offset = safe_margin
    for line in metadata:
        cv2.putText(
            annotation_layer,
            line,
            (x_start, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),  # Red text
            text_thickness
        )
        y_offset += line_spacing

    # If suspicious, annotate suspicious regions
    if suspicious:
        gray = cv2.cvtColor(diff_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 1
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 500:  # Filter out very small regions
                cv2.rectangle(annotation_layer, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text_position = (x, y - 10 if y > 20 else y + 20)
                cv2.putText(
                    annotation_layer,
                    f"NO:{counter}",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),  # Red text
                    text_thickness
                )
                counter += 1

    # Combine annotation layer with the difference-enhanced image
    final_annotated = cv2.addWeighted(diff_cv, 1, annotation_layer, 1, 0)

    # Convert annotated image back to PIL
    annotated_image = Image.fromarray(cv2.cvtColor(final_annotated, cv2.COLOR_BGR2RGB))

    # Enhance brightness, contrast, and saturation of the annotated image
    contrast_enhancer = ImageEnhance.Contrast(annotated_image)
    annotated_image = contrast_enhancer.enhance(1.1)  # Increase contrast

    saturation_enhancer = ImageEnhance.Color(annotated_image)
    annotated_image = saturation_enhancer.enhance(3.2)  # Increase saturation

    # Convert original image to OpenCV format for red-line marking
    original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    # Overlay the same red annotations on the original image
    original_with_annotations = cv2.addWeighted(original_cv, 1, annotation_layer, 1, 0)

    # Convert original with annotations back to PIL
    original_with_annotations = Image.fromarray(cv2.cvtColor(original_with_annotations, cv2.COLOR_BGR2RGB))

    # Stitch the annotated image to the original image
    if direction == 'right':
        new_width = original_with_annotations.width + annotated_image.width
        new_height = max(original_with_annotations.height, annotated_image.height)
        stitched_image = Image.new('RGB', (new_width, new_height))
        stitched_image.paste(original_with_annotations, (0, 0))
        stitched_image.paste(annotated_image, (original_with_annotations.width, 0))
    elif direction == 'bottom':
        new_width = max(original_with_annotations.width, annotated_image.width)
        new_height = original_with_annotations.height + annotated_image.height
        stitched_image = Image.new('RGB', (new_width, new_height))
        stitched_image.paste(original_with_annotations, (0, 0))
        stitched_image.paste(annotated_image, (0, original_with_annotations.height))
    else:
        raise ValueError("Invalid direction. Use 'right' or 'bottom'.")

    # Save the stitched image
    stitched_image.save(output_image_path)
    os.remove(temp_path)



def process_directory(directory, quality=95, scale=10):
    """
    Process all image files in the given directory using ELA for detection and annotation.
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_formats)]

    if not files:
        print("没有找到图片文件。")
        return

    print("六五六五科技数字取证")
    output_dir = os.path.join(directory, "65-Digital-Forensics")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in files:
        input_path = os.path.join(directory, file_name)
        output_path = os.path.join(output_dir, f"ela_{file_name}")

        # Use the previous algorithm to check if the image is suspicious
        is_suspicious = ela_check(input_path, quality=quality)

        # Annotate the image and save the result
        annotate_image(input_path, output_path, scale=scale, suspicious=is_suspicious)

        if is_suspicious:
            print(f"文件名：{file_name} -> 检测到疑似伪造区域")
        else:
            print(f"文件名：{file_name} -> 未检测到疑似伪造区域")

    print(f"\n处理完成，ELA 图像保存路径：{output_dir}")


if __name__ == "__main__":
    current_dir = os.getcwd()
    process_directory(current_dir)