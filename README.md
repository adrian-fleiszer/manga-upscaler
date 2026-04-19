#  Manga Upscaler pipline

Pipeline for donloading and  upscaling manga. This project provides a user-friendly automation for the high-quality upscaling models, making it easy to enhance your images without complex command-line operations.

## Features

- **Versatile Upscaling**: Supports both black & white and color images.
- **Archive Extraction**: Automatically extract images from `.zip` and `.cbz` archives.
- **Dual Interface**: Use the intuitive GUI or the powerful command-line interface (CLI) for automation.
- **Cross-Platform**: Works on Windows, macOS, and Linux.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yui007/manga_upscaler.git
    cd manga_upscaler
    ```

2.  **Install dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate it
    source venv/bin/activate

    # Install the required packages
    pip install -r requirements.txt
    ```


## Usage

### CLI Mode

For automation and scripting, you can use the `manga_upscale.py` script directly.


#### Extract Archives

```bash
python manga_upscale.py extract --input /path/to/your/archives
# Add --overwrite to re-extract existing archives
```

#### Upscale Images

```bash
# Upscale a folder of B&W images
python manga_upscale.py upscale --bw /path/to/bw_images --output /path/to/results

# Upscale a folder of color images
python manga_upscale.py upscale --color /path/to/color_images --output /path/to/results

# Upscale both at once with specific models
python manga_upscale.py upscale \
    --bw /path/to/bw_images \
    --color /path/to/color_images \
    --output /path/to/results \
    --model-bw "4x_MangaJaNai_1200p_V1_ESRGAN_70k.pth" \
    --model-color "4x_IllustrationJaNai_V2standard_DAT2_27k.safetensors"
```


## Acknowledgements & Contributions

This project would not be possible without the incredible work of the following individuals and teams:

-   **Backend Upscaler Engine**: The core upscaling logic is powered by the **[simple_upscaler](https://github.com/thefirst632student/simple_upscaler)** repository by **[thefirst632student](https://github.com/thefirst632student)**. Their work provides the foundation for the powerful image processing in this tool.
-   **High-Quality Models**: Many of the excellent pre-trained models, especially for manga, are from the **[MangaJaNai](https://github.com/the-database/MangaJaNai)** project by **[the-database](https://github.com/the-database)**.

A huge thank you to them for their significant contributions to the open-source community. Please consider supporting their work.

## License

This project is licensed under the MIT License. Note that the included models and backend components may have their own licenses.