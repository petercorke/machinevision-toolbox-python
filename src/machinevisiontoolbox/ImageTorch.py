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
    (``pip install torch`` or
    ``pip install machinevision-toolbox-python[torch]``).
    Each method raises :exc:`ImportError` at call time if PyTorch is not available.
    """

    def tensor(self, device="cpu", normalize=None, dtype=None) -> "torch.Tensor":
        """Convert image to a PyTorch tensor.

        The returned tensor has shape ``(1, C, H, W)`` for multi-plane images
        and ``(1, 1, H, W)`` for single-plane images, with pixel values
        preserved in their original dtype.

        :param device: target device for the tensor (e.g. "cpu", "cuda" or "mps"),
            defaults to "cpu"
        :type device: str, optional
        :param normalize: normalization to apply to pixel values, either
            "imagenet" for standard ImageNet scaling or a tuple of
            (mean, std) lists for custom scaling; if None, no scaling is
            applied and pixel values are preserved in their original range,
            defaults to None
        :type normalize: str or tuple or None, optional
        :param dtype: output tensor dtype, for example ``torch.float32``;
            if None, dtype is inferred from the Image array dtype,
            defaults to None
        :type dtype: torch.dtype or None, optional
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

        Convert to tensor and apply normalization.
        'normalize' can be:
        - None: stays 0.0-1.0 or 0-255
        - "imagenet": applies standard ImageNet mean/std
        - (mean, std): a tuple of lists/arrays, one per channel, for custom scaling
        """
        if not _torch_available:
            raise ImportError(
                "PyTorch is required for to_tensor(). "
                "Install it with: pip install torch "
                "or pip install machinevision-toolbox-python[torch]"
            )
        # 1. Convert to tensor; mono images are (H, W), colour images are (H, W, C)
        array = self.A
        if array.ndim == 2:
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        # 2. Float conversion and normalisation (only when requested)
        if normalize is not None:
            tensor = tensor.float()
            if array.dtype == np.uint8:
                tensor /= 255.0

            if normalize == "imagenet":
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            elif isinstance(normalize, (tuple, list)) and len(normalize) == 2:
                mean, std = normalize
            else:
                mean, std = None, None

            if mean is not None:
                m = torch.tensor(mean).view(-1, 1, 1)
                s = torch.tensor(std).view(-1, 1, 1)
                tensor = (tensor - m) / s

        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        return tensor.unsqueeze(0).to(device)

    @classmethod
    def Tensor(
        cls,
        data,
        logits: bool = False,
        colororder: str | None = None,
        dtype=None,
    ):
        """Create an Image from a PyTorch tensor.

        :param data: tensor input, either a ``torch.Tensor`` of shape
            ``(C, H, W)``, ``(H, W)``, ``(1, C, H, W)`` or ``(1, H, W)``,
            or a dictionary containing a tensor (for example model outputs like
            ``{"out": tensor}``)
        :type data: torch.Tensor or dict
        :param logits: if ``True`` and the tensor is 3D, interpret as class
            logits and apply ``argmax`` over the first axis to create a label
            image, defaults to ``False``
        :type logits: bool, optional
        :param colororder: colour plane order, e.g. ``"RGB"`` or ``"BGR"``,
            defaults to "RGB" for 3-channel images and None for single-channel images
        :type colororder: str, optional
        :param dtype: data type for the image array, e.g. ``np.uint8`` or
            ``np.float32``; if None, the dtype of the input tensor is preserved,
            defaults to same as input tensor dtype
        :type dtype: numpy dtype or None, optional
        :raises ImportError: if PyTorch is not installed
        :raises TypeError: if ``data`` is not a tensor or dictionary containing a tensor
        :raises ValueError: if a 4D tensor has batch size greater than 1
        :return: image wrapping the tensor data
        :rtype: Image

        Accepts either a tensor directly or a dictionary containing a tensor as
        typically returned by torchvision model outputs (for example
        ``{"out": tensor}``).

        Handles:
        - Moving data from GPU/MPS to CPU
        - Detaching from the autograd graph
        - Converting (C, H, W) to (H, W, C)
        - Handling single-image batches ``(B, C, H, W)`` where ``B=1``

        For batches where ``B > 1`` use :class:`TensorStack` which creates an
        image iterator for the batch.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import torch
            >>> tensor = torch.rand(1, 3, 100, 200) # Create a random tensor simulating a model output
            >>> # Convert to Image
            >>> img = Image.Tensor(tensor)
            >>> print(img)

        The modern "Machine Vision Toolbox" workflow::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> outputs = model(img.tensor(normalize="imagenet")) # Pass tensor to model
            >>> out = Image.Tensor(outputs, logits=True).disp()

        :seealso: :meth:`tensor` :class:`TensorStack`
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
        if not isinstance(tensor, torch.Tensor):
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
        if logits:
            # If the user passed raw logits [C, H, W], we take the argmax
            # Note: This is done on the NumPy array here for simplicity,
            # but could be done on the tensor before conversion for speed.
            if x.ndim == 3:
                x = np.argmax(x, axis=0)
            return cls(x)

        # 4. Handle Color/Grayscale Images [C, H, W] -> [H, W, C]
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))

        return cls(x, colororder=colororder, dtype=dtype)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(Path(__file__).parent.parent.parent / "tests" / "test_image_torch.py"),
            "-v",
        ]
    )
