import cv2
import numpy as np


class LightingEstimator:
    def estimate(self, background_image):

        image = self._validate_image(background_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        bright_mask = self._bright_region_mask(hsv)
        direction = self._estimate_direction(luminance, bright_mask)

        bright_pixels = image[bright_mask]
        if bright_pixels.size == 0:
            bright_pixels = image.reshape(-1, 3)

        light_color = bright_pixels.mean(axis=0)[::-1] / 255.0
        ambient_color = image.reshape(-1, 3).mean(axis=0)[::-1] / 255.0
        intensity = float(np.percentile(hsv[:, :, 2], 90) / 255.0)

        return {
            "direction": direction,
            "color": self._to_list(light_color),
            "ambient": self._to_list(ambient_color),
            "intensity": intensity,
        }

    def _validate_image(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("background_image debe ser una imagen BGR de OpenCV")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("background_image debe tener tres canales BGR")
        if image.size == 0:
            raise ValueError("background_image no puede estar vacia")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _bright_region_mask(self, hsv):
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        bright_threshold = np.percentile(value, 85)
        bright_mask = value >= bright_threshold

        low_saturation = saturation <= np.percentile(saturation, 70)
        neutral_bright_mask = bright_mask & low_saturation
        if np.count_nonzero(neutral_bright_mask) >= max(1, value.size // 100):
            bright_mask = neutral_bright_mask

        return bright_mask

    def _estimate_direction(self, luminance, bright_mask):
        height, width = luminance.shape
        y_coordinates, x_coordinates = np.indices(luminance.shape)
        weights = luminance * bright_mask
        total_weight = weights.sum()

        if total_weight > 0:
            center_x = float((x_coordinates * weights).sum() / total_weight)
            center_y = float((y_coordinates * weights).sum() / total_weight)
        else:
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0

        direction = np.array([
            center_x - (width - 1) / 2.0,
            center_y - (height - 1) / 2.0,
        ], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        return self._to_list(direction)

    def _to_list(self, values):
        return [float(value) for value in values]


if __name__ == "__main__":
    image = cv2.imread("data/background/sunset.jpg")
    print(LightingEstimator().estimate(image))