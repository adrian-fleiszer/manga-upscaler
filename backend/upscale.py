#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
# Add the backend directory to the Python path to ensure robust imports
sys.path.append(str(Path(__file__).parent.parent))
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import typer
from rich import print
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TaskID, TimeRemainingColumn
import safetensors.torch

import utils.dataops as ops
from utils.architecture.RRDB import RRDBNet as ESRGAN
from utils.architecture.SPSR import SPSRNet as SPSR
from utils.architecture.SRVGG import SRVGGNetCompact as RealESRGANv2
from utils.architecture.FDAT import FDATNet as FDAT
from utils.architecture.DAT import DATNet as DAT
from utils.architecture.DAT_variants import detect_dat_variant, DAT_CONFIGS


class SeamlessOptions(str, Enum):
    TILE = "tile"
    MIRROR = "mirror"
    REPLICATE = "replicate"
    ALPHA_PAD = "alpha_pad"


class AlphaOptions(str, Enum):
    NO_ALPHA = "none"
    BG_DIFFERENCE = "bg_difference"
    ALPHA_SEPARATELY = "separate"
    SWAPPING = "swapping"


class Upscale:
    model_str: Optional[str] = None
    input: Optional[Path] = None
    output: Optional[Path] = None
    reverse: Optional[bool] = None
    skip_existing: Optional[bool] = None
    delete_input: Optional[bool] = None
    seamless: Optional[SeamlessOptions] = None
    cpu: Optional[bool] = None
    fp16: Optional[bool] = None
    # device_id: int = None
    cache_max_split_depth: Optional[bool] = None
    binary_alpha: Optional[bool] = None
    ternary_alpha: Optional[bool] = None
    alpha_threshold: Optional[float] = None
    alpha_boundary_offset: Optional[float] = None
    alpha_mode: Optional[AlphaOptions] = None
    log: Optional[logging.Logger] = None

    device: Optional[torch.device] = None
    in_nc: Optional[int] = None
    out_nc: Optional[int] = None
    last_model: Optional[str] = None
    last_in_nc: Optional[int] = None
    last_out_nc: Optional[int] = None
    last_nf: Optional[int] = None
    last_nb: Optional[int] = None
    last_scale: Optional[int] = None
    last_kind: Optional[str] = None
    model: Optional[Union[torch.nn.Module, ESRGAN, RealESRGANv2, SPSR, FDAT]] = None

    def __init__(
        self,
        model: str,
        input: Path,
        output: Path,
        reverse: bool = False,
        skip_existing: bool = False,
        delete_input: bool = False,
        seamless: Optional[SeamlessOptions] = None,
        cpu: bool = False,
        fp16: bool = False,
        device_id: int = 0,
        cache_max_split_depth: bool = False,
        binary_alpha: bool = False,
        ternary_alpha: bool = False,
        alpha_threshold: float = 0.5,
        alpha_boundary_offset: float = 0.2,
        alpha_mode: Optional[AlphaOptions] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        self.model_str = model
        self.input = input.resolve()
        self.output = output.resolve()
        self.reverse = reverse
        self.skip_existing = skip_existing
        self.delete_input = delete_input
        self.seamless = seamless
        self.cpu = cpu
        self.fp16 = fp16
        self.device = torch.device("cpu" if self.cpu else f"cuda:{device_id}")
        self.cache_max_split_depth = cache_max_split_depth
        self.binary_alpha = binary_alpha
        self.ternary_alpha = ternary_alpha
        self.alpha_threshold = alpha_threshold
        self.alpha_boundary_offset = alpha_boundary_offset
        self.alpha_mode = alpha_mode
        self.log = log
        if self.fp16:
            torch.set_default_dtype(torch.float16)

    def run(self) -> None:
        if not self.model_str or not self.input or not self.output or not self.log:
            print("Error: Core attributes (model_str, input, output, log) are not initialized.")
            return

        model_chain = (
            self.model_str.split("+")
            if "+" in self.model_str
            else self.model_str.split(">")
        )

        for idx, model in enumerate(model_chain):
            interpolations = (
                model.split("|") if "|" in self.model_str else model.split("&")
            )

            if len(interpolations) > 1:
                for i, interpolation in enumerate(interpolations):
                    interp_model, interp_amount = (
                        interpolation.split("@")
                        if "@" in interpolation
                        else interpolation.split(":")
                    )
                    interp_model = self.__check_model_path(interp_model)
                    interpolations[i] = f"{interp_model}@{interp_amount}"
                model_chain[idx] = "&".join(interpolations)
            else:
                model_chain[idx] = self.__check_model_path(model)

        if not self.input.exists():
            self.log.error(f'Folder "{self.input}" does not exist.')
            sys.exit(1)
        elif self.input.is_file():
            self.log.error(f'Folder "{self.input}" is a file.')
            sys.exit(1)
        elif self.output.is_file():
            self.log.error(f'Folder "{self.output}" is a file.')
            sys.exit(1)
        elif not self.output.exists():
            self.output.mkdir(parents=True)

        print(
            'Model{:s}: "{:s}"'.format(
                "s" if len(model_chain) > 1 else "",
                # ", ".join([Path(x).stem for x in model_chain]),
                ", ".join([x for x in model_chain]),
            )
        )

        images: List[Path] = []
        # List of extensions: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
        # Also gif and tga which seem to be supported as well though are undocumented.

        # Now ESRGAN doesn't ignore files with extensions like .JPG
        extensions = [
            "bmp",
            "dib",
            "jpeg",
            "jpg",
            "jpe",
            "jp2",
            "png",
            "webp",
            "pbm",
            "pgm",
            "ppm",
            "pxm",
            "pnm",
            "pfm",
            "sr",
            "ras",
            "tiff",
            "tif",
            "exr",
            "hdr",
            "pic",
            "gif",
            "tga",
        ]

        for file in self.input.glob("**/*.*"):
            if file.suffix.lower()[1:] in extensions:
                images.append(file)

        # Store the maximum split depths for each model in the chain
        # TODO: there might be a better way of doing this but it's good enough for now
        split_depths = {}

        with Progress(
            # SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            task_upscaling = progress.add_task("Upscaling", total=len(images))
            for idx, img_path in enumerate(images, 1):
                img_input_path_rel = img_path.relative_to(self.input)
                if not self.output: continue
                output_dir = self.output.joinpath(img_input_path_rel).parent
                img_output_path_rel = output_dir.joinpath(f"{img_path.stem}.jpg")
                output_dir.mkdir(parents=True, exist_ok=True)
                if len(model_chain) == 1:
                    self.log.info(
                        f'Processing {str(idx).zfill(len(str(len(images))))}: "{img_input_path_rel}"'
                    )
                if self.skip_existing and img_output_path_rel.is_file():
                    self.log.warning("Already exists, skipping")
                    if self.delete_input:
                        img_path.unlink(missing_ok=True)
                    progress.advance(task_upscaling)
                    continue
                # read image
                # We use imdecode instead of imread to work around Unicode breakage on Windows.
                # See https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
                img = cv2.imdecode(
                    np.fromfile(str(img_path.absolute()), dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                if img is None:
                    self.log.warning(f"Could not read image: {img_path}, skipping.")
                    progress.advance(task_upscaling)
                    continue
                if len(img.shape) < 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Seamless modes
                if self.seamless == SeamlessOptions.TILE:
                    img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
                elif self.seamless == SeamlessOptions.MIRROR:
                    img = cv2.copyMakeBorder(
                        img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101
                    )
                elif self.seamless == SeamlessOptions.REPLICATE:
                    img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
                elif self.seamless == SeamlessOptions.ALPHA_PAD:
                    img = cv2.copyMakeBorder(
                        img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
                    )
                final_scale: int = 1

                task_model_chain: Optional[TaskID] = None
                if len(model_chain) > 1:
                    task_model_chain = progress.add_task(
                        f'{str(idx).zfill(len(str(len(images))))} - "{img_input_path_rel}"',
                        total=len(model_chain),
                    )
                rlt = None # Initialize rlt
                for i, model_path in enumerate(model_chain):
                    img_height, img_width = img.shape[:2]

                    # Load the model so we can access the scale
                    self.load_model(model_path)

                    if self.last_scale is None:
                        if self.log: self.log.error("Error: Model scale was not determined.")
                        continue
                    if self.cache_max_split_depth and len(split_depths.keys()) > 0:
                        rlt, depth = ops.auto_split_upscale(
                            img,
                            self.upscale,
                            self.last_scale,
                            max_depth=split_depths[i],
                        )
                    else:
                        rlt, depth = ops.auto_split_upscale(
                            img, self.upscale, self.last_scale
                        )
                        split_depths[i] = depth

                    final_scale *= self.last_scale

                    # This is for model chaining
                    img = rlt.astype("uint8")
                    if len(model_chain) > 1 and task_model_chain is not None:
                        progress.advance(task_model_chain)

                if self.seamless and rlt is not None:
                    rlt = self.crop_seamless(rlt, final_scale)

                # We use imencode instead of imwrite to work around Unicode breakage on Windows.
                # See https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
                if rlt is None:
                    self.log.error(f"Upscaling failed for {img_path}, result is None.")
                    continue
                is_success, im_buf_arr = cv2.imencode(".jpg", rlt, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not is_success:
                    raise Exception("cv2.imencode failure")
                im_buf_arr.tofile(str(img_output_path_rel.absolute()))

                if self.delete_input:
                    img_path.unlink(missing_ok=True)

                progress.advance(task_upscaling)

    def __check_model_path(self, model_path: str) -> str:
        if Path(model_path).is_file():
            return model_path
        elif Path("./models/").joinpath(model_path).is_file():
            return str(Path("./models/").joinpath(model_path))
        else:
            if self.log:
                self.log.error(f'Model "{model_path}" does not exist.')
            sys.exit(1)

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def process(self, img: np.ndarray):
        """
        Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

                Parameters:
                        img (array): The image to process

                Returns:
                        rlt (array): The processed image
        """
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 4:
            img = img[:, :, [2, 1, 0, 3]]
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        if self.fp16:
            img_tensor = img_tensor.half()
        img_LR = img_tensor.unsqueeze(0)
        if self.device:
            img_LR = img_LR.to(self.device)

        if not self.model:
             raise Exception("Model not loaded")
        output = self.model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        if output.shape[0] == 3:
            output = output[[2, 1, 0], :, :]
        elif output.shape[0] == 4:
            output = output[[2, 1, 0, 3], :, :]
        output = np.transpose(output, (1, 2, 0))
        return output

    def load_model(self, model_path: str):
        if model_path != self.last_model:
            # interpolating OTF, example: 4xBox:25&4xPSNR:75
            if (":" in model_path or "@" in model_path) and (
                "&" in model_path or "|" in model_path
            ):
                interps = model_path.split("&")[:2]
                model_1_path = interps[0].split("@")[0]
                model_2_path = interps[1].split("@")[0]
                
                # Load each model using appropriate method based on file extension
                model_1 = self._load_state_dict(model_1_path)
                model_2 = self._load_state_dict(model_2_path)
                
                state_dict = OrderedDict()
                for k, v_1 in model_1.items():
                    v_2 = model_2[k]
                    state_dict[k] = (int(interps[0].split("@")[1]) / 100) * v_1 + (
                        int(interps[1].split("@")[1]) / 100
                    ) * v_2
            else:
                state_dict = self._load_state_dict(model_path)

            # SRVGGNet Real-ESRGAN (v2)
            if (
                "params" in state_dict
                and "body.0.weight" in state_dict["params"]
            ):
                self.model = RealESRGANv2(state_dict)
                self.last_in_nc = self.model.num_in_ch
                self.last_out_nc = self.model.num_out_ch
                self.last_nf = self.model.num_feat
                self.last_nb = self.model.num_conv
                self.last_scale = self.model.scale
                self.last_model = model_path
                self.last_kind = "RealESRGAN-v2"
            # FDAT (Fast Dual Aggregation Transformer)
            elif "conv_first.weight" in state_dict and "groups.0.blocks.0.n1.weight" in state_dict:
                self.model = FDAT(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_feat
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path
                self.last_kind = "FDAT"
            # SPSR (ESRGAN with lots of extra layers)
            elif "f_HR_conv1.0.weight" in state_dict:
                self.model = SPSR(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_filters
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path
                self.last_kind = "SPSR"
            # DAT (Dual Aggregation Transformer) and variants
            elif self._is_dat_architecture(state_dict):
                dat_variant = detect_dat_variant(state_dict)
                self.model = DAT(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_feat
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path
                self.last_kind = f"DAT ({dat_variant})"
            # Check for unsupported architectures
            elif self._is_unsupported_architecture(state_dict):
                unsupported_type = self._detect_unsupported_architecture(state_dict)
                if self.log:
                    self.log.error(f'Unsupported model architecture "{unsupported_type}" in model "{model_path}".')
                    self.log.error('This model architecture is not currently supported by this upscaler.')
                    if unsupported_type == "DAT2":
                        self.log.error('DAT2 models require a different architecture implementation.')
                sys.exit(1)
            # Regular ESRGAN, "new-arch" ESRGAN, Real-ESRGAN v1
            else:
                self.model = ESRGAN(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_filters
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path
                self.last_kind = "ESRGAN"

            del state_dict
        if self.model:
            self.model.eval()
            for k, v in self.model.named_parameters():
                v.requires_grad = False
            if self.device:
                self.model = self.model.to(self.device)
        self.last_model = model_path

    def _load_state_dict(self, model_path: str):
        """
        Load a state dictionary from either a .pth or .safetensors file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            dict: The loaded state dictionary
        """
        model_path_obj = Path(model_path)
        
        if model_path_obj.suffix.lower() == '.safetensors':
            # Load safetensors file
            return safetensors.torch.load_file(model_path)
        else:
            # Load PyTorch file (.pth, .pt, etc.)
            # Map to CPU if using CPU mode
            map_location = "cpu" if self.cpu else None
            return torch.load(model_path, weights_only=False, map_location=map_location)

    def _is_dat_architecture(self, state_dict):
        """
        Check if the model uses DAT (Dual Aggregation Transformer) architecture.
        
        Args:
            state_dict (dict): The model state dictionary
            
        Returns:
            bool: True if the architecture is DAT
        """
        # DAT models have these characteristic keys:
        # - conv_first layer
        # - layers.X.blocks.Y pattern (residual groups with blocks)
        # - before_RG layer (before residual groups)
        # - attn modules in blocks
        
        has_conv_first = 'conv_first.weight' in state_dict
        has_layers = any('layers.' in key for key in state_dict.keys())
        has_blocks = any('blocks.' in key for key in state_dict.keys()) 
        has_before_rg = any('before_RG' in key for key in state_dict.keys())
        has_attn = any('attn.' in key for key in state_dict.keys())
        
        # Check for the specific DAT pattern: layers.X.blocks.Y.attn structure
        has_dat_pattern = False
        for key in state_dict.keys():
            if 'layers.' in key and 'blocks.' in key and 'attn.' in key:
                has_dat_pattern = True
                break
        
        # DAT should have conv_first, layers with blocks, and attention modules
        return has_conv_first and has_layers and has_blocks and has_attn and has_dat_pattern

    def _is_unsupported_architecture(self, state_dict):
        """
        Check if the model uses an unsupported architecture.
        
        Args:
            state_dict (dict): The model state dictionary
            
        Returns:
            bool: True if the architecture is unsupported
        """
        # Since we now support DAT, we need to check for other unsupported patterns
        # For now, we'll return False as we support most common architectures
        # This can be extended in the future for truly unsupported architectures
        return False
    
    def _detect_unsupported_architecture(self, state_dict):
        """
        Detect the specific unsupported architecture type.
        
        Args:
            state_dict (dict): The model state dictionary
            
        Returns:
            str: The architecture type name
        """
        # Check for DAT2 pattern
        sample_keys = list(state_dict.keys())[:10]
        for key in sample_keys:
            if "layers." in key and "blocks." in key and "attn.attns." in key:
                return "DAT2"
        
        return "Unknown"

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscales the image passed in with the specified model

                Parameters:
                        img: The image to upscale
                        model_path (string): The model to use

                Returns:
                        output: The processed image
        """

        img = img * 1.0 / np.iinfo(img.dtype).max

        if (
            img.ndim == 3
            and img.shape[2] == 4
            and self.last_in_nc == 3
            and self.last_out_nc == 3
        ):
            # Fill alpha with white and with black, remove the difference
            if self.alpha_mode == AlphaOptions.BG_DIFFERENCE:
                img1 = np.copy(img[:, :, :3])
                img2 = np.copy(img[:, :, :3])
                for c in range(3):
                    img1[:, :, c] *= img[:, :, 3]
                    img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

                output1 = self.process(img1)
                output2 = self.process(img2)
                alpha = 1 - np.mean(output2 - output1, axis=2)
                output = np.dstack((output1, alpha))
                output = np.clip(output, 0, 1)
            # Upscale the alpha channel itself as its own image
            elif self.alpha_mode == AlphaOptions.ALPHA_SEPARATELY:
                img1 = np.copy(img[:, :, :3])
                img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 0],
                    )
                )
            # Use the alpha channel like a regular channel
            elif self.alpha_mode == AlphaOptions.SWAPPING:
                img1 = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
                img2 = cv2.merge((img[:, :, 1], img[:, :, 2], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 2],
                    )
                )
            # Remove alpha
            else:
                img1 = np.copy(img[:, :, :3])
                output = self.process(img1)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)

            if self.binary_alpha:
                alpha = output[:, :, 3]
                threshold = self.alpha_threshold if self.alpha_threshold is not None else 0.5
                _, alpha = cv2.threshold(alpha, threshold, 1, cv2.THRESH_BINARY)
                output[:, :, 3] = alpha
            elif self.ternary_alpha:
                alpha = output[:, :, 3]
                threshold = self.alpha_threshold if self.alpha_threshold is not None else 0.5
                offset = self.alpha_boundary_offset if self.alpha_boundary_offset is not None else 0.2
                half_transparent_lower_bound = (
                    threshold - offset
                )
                half_transparent_upper_bound = (
                    threshold + offset
                )
                alpha = np.where(
                    alpha < half_transparent_lower_bound,
                    0,
                    np.where(alpha <= half_transparent_upper_bound, 0.5, 1),
                )
                output[:, :, 3] = alpha
        else:
            if img.ndim == 2:
                last_in_nc = self.last_in_nc if self.last_in_nc is not None else 3
                img = np.tile(
                    np.expand_dims(img, axis=2), (1, 1, min(last_in_nc, 3))
                )
            if self.last_in_nc and img.shape[2] > self.last_in_nc:  # remove extra channels
                if self.log: self.log.warning("Truncating image channels")
                img = img[:, :, : self.last_in_nc]
            # pad with solid alpha channel
            elif self.last_in_nc and img.shape[2] == 3 and self.last_in_nc == 4:
                img = np.dstack((img, np.full(img.shape[:-1], 1.0)))
            output = self.process(img)

        output = (output * 255.0).round()

        return output

    def crop_seamless(self, img: np.ndarray, scale: int) -> np.ndarray:
        img_height, img_width = img.shape[:2]
        y, x = 16 * scale, 16 * scale
        h, w = img_height - (32 * scale), img_width - (32 * scale)
        img = img[y : y + h, x : x + w]
        return img


app = typer.Typer()


@app.command()
def main(
    model: str = typer.Argument(...),
    input: Path = typer.Option(Path("input"), "--input", "-i", help="Input folder"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output folder"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse Order"),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        "-se",
        help="Skip existing output files",
    ),
    delete_input: bool = typer.Option(
        False,
        "--delete-input",
        "-di",
        help="Delete input files after upscaling",
    ),
    seamless: SeamlessOptions = typer.Option(
        None,
        "--seamless",
        "-s",
        case_sensitive=False,
        help="Helps seamlessly upscale an image. tile = repeating along edges. mirror = reflected along edges. replicate = extended pixels along edges. alpha_pad = extended alpha border.",
    ),
    cpu: bool = typer.Option(False, "--cpu", "-c", help="Use CPU instead of CUDA"),
    fp16: bool = typer.Option(
        True,
        "--floating-point-16",
        "-fp16",
        help="Use FloatingPoint16/Halftensor type for images.",
    ),
    device_id: int = typer.Option(
        0, "--device-id", "-did", help="The numerical ID of the GPU you want to use."
    ),
    cache_max_split_depth: bool = typer.Option(
        False,
        "--cache-max-split-depth",
        "-cmsd",
        help="Caches the maximum recursion depth used by the split/merge function. Useful only when upscaling images of the same size.",
    ),
    binary_alpha: bool = typer.Option(
        False,
        "--binary-alpha",
        "-ba",
        help="Whether to use a 1 bit alpha transparency channel, Useful for PSX upscaling",
    ),
    ternary_alpha: bool = typer.Option(
        False,
        "--ternary-alpha",
        "-ta",
        help="Whether to use a 2 bit alpha transparency channel, Useful for PSX upscaling",
    ),
    alpha_threshold: float = typer.Option(
        0.5,
        "--alpha-threshold",
        "-at",
        help="Only used when binary_alpha is supplied. Defines the alpha threshold for binary transparency",
    ),
    alpha_boundary_offset: float = typer.Option(
        0.2,
        "--alpha-boundary-offset",
        "-abo",
        help="Only used when binary_alpha is supplied. Determines the offset boundary from the alpha threshold for half transparency.",
    ),
    alpha_mode: AlphaOptions = typer.Option(
        None,
        "--alpha-mode",
        "-am",
        help="Type of alpha processing to use. no_alpha = is no alpha processing. bas = is BA's difference method. alpha_separately = is upscaling the alpha channel separately (like IEU). swapping = is swapping an existing channel with the alpha channel.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose mode",
    ),
):
    logging.basicConfig(
        # On Google Colab additional messages override the progressbar, so now
        # the logging level by default is set to ERROR.
        level=logging.DEBUG if verbose else logging.ERROR,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
        # handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )

    upscale = Upscale(
        model=model,
        input=input,
        output=output,
        reverse=reverse,
        skip_existing=skip_existing,
        delete_input=delete_input,
        seamless=seamless,
        cpu=cpu,
        fp16=fp16,
        device_id=device_id,
        cache_max_split_depth=cache_max_split_depth,
        binary_alpha=binary_alpha,
        ternary_alpha=ternary_alpha,
        alpha_threshold=alpha_threshold,
        alpha_boundary_offset=alpha_boundary_offset,
        alpha_mode=alpha_mode,
    )
    upscale.run()


if __name__ == "__main__":
    app()
