自动标注潜在风险并评估图片是否伪造，用于初步筛查
# Image Tampering Detection

This project provides an image tampering detection tool using Error Level Analysis (ELA). The tool can analyze images for potential tampering and annotate suspicious regions.

## Features

- **Error Level Analysis (ELA)**: Detects differences between original and recompressed images to identify potential tampering.
- **Automatic Annotation**: Annotates suspicious regions in the image and provides metadata.
- **Batch Processing**: Processes all images in a directory and saves the results.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/l6565kj/Image-tampering-detection.git
    cd Image-tampering-detection
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place the images you want to analyze in the current directory or specify a directory containing the images.

2. Run the script:
    ```sh
    python 65-Digital-Forensics.py
    ```

3. The script will process all supported image files in the specified directory and save the annotated results in a subdirectory named `65-Digital-Forensics`.

## Supported Image Formats

- JPEG
- PNG
- BMP
- TIFF

## Example

```sh
python 65-Digital-Forensics.py
