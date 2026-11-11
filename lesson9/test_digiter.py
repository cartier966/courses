import base64
import io
import unittest

import numpy as np
from PIL import Image, ImageDraw

from digiter import load_model, preprocess_image


def image_url(box=None):
    image = Image.new("RGBA", (280, 280))
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class DigitRecognizerTests(unittest.TestCase):
    def test_blank_image(self):
        pixels = preprocess_image(image_url())
        self.assertEqual(pixels.shape, (28, 28))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertFalse(np.any(pixels))

    def test_corner_drawing_is_centered_and_normalized(self):
        centered = preprocess_image(image_url((100, 100, 139, 199)))
        corner = preprocess_image(image_url((0, 0, 39, 99)))
        np.testing.assert_array_equal(centered, corner)
        self.assertGreaterEqual(float(corner.min()), 0)
        self.assertLessEqual(float(corner.max()), 1)
        rows, columns = np.nonzero(corner)
        self.assertGreaterEqual(rows.min(), 4)
        self.assertLessEqual(rows.max(), 23)
        self.assertGreaterEqual(columns.min(), 4)
        self.assertLessEqual(columns.max(), 23)

    def test_saved_model_returns_ten_probabilities(self):
        pixels = preprocess_image(image_url((100, 100, 139, 199)))
        probabilities = np.asarray(load_model()(pixels.reshape(1, 28, 28, 1), training=False))
        self.assertEqual(probabilities.shape, (1, 10))
        self.assertTrue(np.all(probabilities >= 0))
        np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-5)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_image("data:image/jpeg;base64,abc")


if __name__ == "__main__":
    unittest.main()
