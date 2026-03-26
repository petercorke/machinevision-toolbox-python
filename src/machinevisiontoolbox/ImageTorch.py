"""
PyTorch tensor conversion and integration for Image objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machinevisiontoolbox.ImageCore import Image

import numpy as np

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False


# TODO need a resizing option for the torch() method, to allow scaling to a fixed size for model input


class ImageTorchMixin:
    """
    PyTorch integration methods for the Image class.

    Methods in this mixin require PyTorch to be installed
    (``pip install torch``).  Each method raises :exc:`ImportError` at
    call time if PyTorch is not available.
    """

    def torch(self, device="cpu", normalize="imagenet") -> "torch.Tensor":
        """Convert image to a PyTorch tensor.

        The returned tensor has shape ``(1, C, H, W)`` for multi-plane images
        and ``(1, 1, H, W)`` for single-plane images, with pixel values
        preserved in their original dtype.

        :param device: target device for the tensor (e.g. "cpu" or "cuda"),
            defaults to "cpu"
        :type device: str, optional
        :param normalize: normalization to apply to pixel values, either
            "imagenet" for standard ImageNet scaling or a tuple of
            (mean, std) lists for custom scaling; if None, no scaling is
            applied and pixel values are preserved in their original range,
            defaults to "imagenet"
        :type normalize: str or tuple or None, optional
        :raises ImportError: if PyTorch is not installed
        :return: image as a PyTorch tensor
        :rtype: torch.Tensor

        .. note:: Pixel values are *not* normalised by default; set
            ``normalize="imagenet"`` or provide custom mean/std to scale to
            zero mean and unit variance if required for model input.


        The returned tensor has shape ``(C, H, W)`` for multi-plane images
        and ``(1, H, W)`` for single-plane images, with pixel values
        preserved in their original dtype.

        :raises ImportError: if PyTorch is not installed
        :return: image as a PyTorch tensor
        :rtype: torch.Tensor

        .. note:: Pixel values are *not* normalised; scale to ``[0, 1]``
            manually if required for model input.
        """
        if not _torch_available:
            raise ImportError(
                "PyTorch is required for to_tensor(). "
                "Install it with: pip install torch"
            )
        """
        Convert to tensor and apply normalization.
        'normalize' can be:
        - None: stays 0-1 or 0-255
        - "imagenet": applies standard ImageNet mean/std
        - (mean, std): a tuple of lists/arrays for custom scaling
        """
        # 1. Basic conversion to float 0.0 - 1.0
        tensor = torch.from_numpy(self.data).permute(2, 0, 1).float()
        if self.data.dtype == np.uint8:
            tensor /= 255.0

        # 2. Handle Normalization Presets
        if normalize == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif isinstance(normalize, (tuple, list)) and len(normalize) == 2:
            mean, std = normalize
        else:
            mean, std = None, None

        # 3. Apply the Transformation: (x - mean) / std
        if mean is not None:
            # Convert mean/std to tensors and reshape for broadcasting: [C, 1, 1]
            m = torch.tensor(mean).view(-1, 1, 1)
            s = torch.tensor(std).view(-1, 1, 1)
            tensor = (tensor - m) / s

        return tensor.unsqueeze(0).to(device)

    @classmethod
    def Torch(cls, data, is_mask=False, colororder: str | None = None):
        """Create an Image from a PyTorch tensor.

        :param tensor: tensor of shape ``(C, H, W)`` or ``(H, W)``
        :type tensor: torch.Tensor
        :param colororder: colour plane order, e.g. ``"RGB"`` or ``"BGR"``,
            defaults to None
        :type colororder: str, optional
        :raises ImportError: if PyTorch is not installed
        :return: image wrapping the tensor data
        :rtype: Image

        Accepts either a torch.Tensor or a dictionary containing a tensor.
        Create an Image object from a PyTorch tensor.

        Handles:
        - Moving data from GPU/MPS to CPU
        - Detaching from the autograd graph
        - Converting (C, H, W) to (H, W, C)
        - Handling single-image batches (B, C, H, W)


        # Modern "Machine Vision Toolbox" workflow
        outputs = model(img.torch())
        out = Image.torch(outputs, is_mask=True).disp()

        """
        # 1. Handle Dictionary Input (The "Friendly" check)
        if isinstance(data, dict):
            # Look for the standard 'out' key, otherwise grab the first value
            tensor = data.get("out")
            if tensor is None:
                tensor = next(iter(data.values()))
        else:
            tensor = data

        # 2. Validation
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected torch.Tensor or dict, got {type(data)}")

        # 1. Boilerplate: Detach, Move to CPU, and convert to NumPy
        # We use .float() or .byte() depending on the need, but usually
        # keeping the original dtype is safest.
        x = tensor.detach().cpu().numpy()

        # 2. Handle Batch dimension [B, C, H, W] -> [C, H, W]
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = np.squeeze(x, axis=0)
            else:
                raise ValueError(
                    f"Expected a single image batch, but got shape {x.shape}"
                )

        # 3. Handle Semantic Segmentation Masks (Argmax case)
        if is_mask:
            # If the user passed raw logits [C, H, W], we take the argmax
            # Note: This is done on the NumPy array here for simplicity,
            # but could be done on the tensor before conversion for speed.
            if x.ndim == 3:
                x = np.argmax(x, axis=0)
            return cls(x)

        # 4. Handle Color/Grayscale Images [C, H, W] -> [H, W, C]
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))

        return cls(x)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(Path(__file__).parent.parent.parent / "tests" / "test_image_torch.py"),
            "-v",
        ]
    )
