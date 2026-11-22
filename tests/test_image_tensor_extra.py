#!/usr/bin/env python
"""More tests for ``ImageTensor``: normalisation, dictionary input and errors."""

import numpy as np
import numpy.testing as nt
import pytest

from machinevisiontoolbox import Image
from machinevisiontoolbox import ImageTensor as imt

torch = pytest.importorskip("torch")


def test_tensor_needs_torch(monkeypatch):
    monkeypatch.setattr(imt, "_torch_available", False)
    with pytest.raises(ImportError, match="PyTorch is required"):
        Image.Zeros(size=(4, 4), dtype="uint8").tensor()


class TestNormalize:
    @pytest.fixture
    def img(self):
        # every pixel is (255, 0, 51)
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        arr[:, :, 2] = 51
        return Image(arr, colororder="RGB")

    def test_imagenet(self, img):
        t = img.tensor(normalize="imagenet")
        assert t.shape == (1, 3, 4, 4)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        expected = [
            (1.0 - mean[0]) / std[0],
            (0 - mean[1]) / std[1],
            (0.2 - mean[2]) / std[2],
        ]
        nt.assert_allclose(t[0, :, 0, 0].numpy(), expected, rtol=1e-5)

    def test_custom_mean_and_std(self, img):
        t = img.tensor(normalize=([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]))
        nt.assert_allclose(t[0, :, 0, 0].numpy(), [2.0, -2.0, -1.2], rtol=1e-5)

    def test_custom_as_a_list(self, img):
        t = img.tensor(normalize=[[0, 0, 0], [1, 1, 1]])
        # scaled to 0..1 only
        nt.assert_allclose(t[0, :, 0, 0].numpy(), [1.0, 0.0, 0.2], rtol=1e-5)

    def test_unrecognised_normalize_just_scales(self, img):
        t = img.tensor(normalize="other")
        assert t.dtype == torch.float32
        nt.assert_allclose(t[0, :, 0, 0].numpy(), [1.0, 0.0, 0.2], rtol=1e-5)

    def test_float_image_is_not_rescaled(self):
        im = Image(np.full((3, 3), 0.5, dtype=np.float32))
        t = im.tensor(normalize=([0.0], [1.0]))
        nt.assert_allclose(t.numpy(), 0.5)


class TestFromTensor:
    def test_dictionary_uses_the_out_key(self):
        x = torch.zeros((1, 3, 4, 4))
        x[0, 0] = 1
        img = Image.Tensor({"out": x, "aux": torch.ones(1, 3, 2, 2)}, colororder="RGB")
        assert img.shape == (4, 4, 3)
        assert img.array[:, :, 0].min() == 1

    def test_dictionary_without_out_uses_the_first_value(self):
        x = torch.ones((1, 3, 5, 5))
        img = Image.Tensor(
            {"logits": x, "other": torch.zeros(1, 3, 2, 2)}, colororder="RGB"
        )
        assert img.shape == (5, 5, 3)

    def test_not_a_tensor(self):
        with pytest.raises(TypeError, match="Expected torch.Tensor or dict"):
            Image.Tensor(np.zeros((3, 4, 4)))

    def test_batch_of_one_is_squeezed(self):
        img = Image.Tensor(torch.ones((1, 3, 4, 6)), colororder="RGB")
        assert img.shape == (4, 6, 3)

    def test_batch_of_many_is_an_error(self):
        with pytest.raises(ValueError, match="single image batch"):
            Image.Tensor(torch.ones((2, 3, 4, 4)))

    def test_two_dimensional_tensor_is_a_mono_image(self):
        img = Image.Tensor(torch.arange(12.0).reshape(3, 4))
        assert img.shape == (3, 4)

    def test_logits_two_dimensional_is_passed_through(self):
        img = Image.Tensor(torch.arange(6.0).reshape(2, 3), logits=True)
        assert img.shape == (2, 3)

    def test_tensor_that_requires_grad(self):
        x = torch.ones((3, 4, 4), requires_grad=True)
        img = Image.Tensor(x, colororder="RGB")
        assert img.shape == (4, 4, 3)

    def test_roundtrip_with_normalisation_is_not_exact_but_close(self):
        arr = np.random.default_rng(0).integers(0, 255, (5, 6, 3), dtype=np.uint8)
        img = Image(arr, colororder="RGB")
        t = img.tensor(normalize=([0, 0, 0], [1, 1, 1]))
        back = Image.Tensor(t, colororder="RGB")
        nt.assert_allclose(back.array * 255, arr, atol=1e-3)
