import cv2
import numpy as np


class LightingEstimator:
    #Estimar parametros basicos de iluminacion usando reglas de vision clasica.

    #La clase no intenta reconstruir la iluminacion 3D de forma exacta. En sulugar, busca las zonas mas luminosas de la imagen y las usa como una
    #aproximacion de la fuente de luz.

    def estimate(self, background_image):
        #Devuelve direccion, color, ambiente e intensidad de una imagen BGR.
        # HSV permite separar el brillo de la saturacion del color.
        image = self._validate_image(background_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Las zonas mas brillantes se usan como aproximacion de la fuente de luz.
        bright_mask = self._bright_region_mask(hsv)
        direction = self._estimate_direction(luminance, bright_mask)

        bright_pixels = image[bright_mask]
        if bright_pixels.size == 0:
            # Se usa toda la imagen como respaldo si no hay zonas seleccionadas.
            bright_pixels = image.reshape(-1, 3)

        # OpenCV usa BGR; la salida se expresa en RGB normalizado.
        light_color = bright_pixels.mean(axis=0)[::-1] / 255.0
        # El promedio global representa el color base o ambiente de la escena.
        ambient_color = image.reshape(-1, 3).mean(axis=0)[::-1] / 255.0

        # Un percentil alto evita que las zonas oscuras dominen la intensidad.
        intensity = float(np.percentile(hsv[:, :, 2], 90) / 255.0)

        return {
            "direction": direction,
            "color": self._to_list(light_color),
            "ambient": self._to_list(ambient_color),
            "intensity": intensity,
        }

    def _validate_image(self, image):
        #Comprueba el formato esperado y normaliza el tipo de los pixeles.
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
        #Crea una mascara de pixeles brillantes y poco saturados.
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        # El umbral relativo se adapta a escenas claras u oscuras.
        bright_threshold = np.percentile(value, 85)
        bright_mask = value >= bright_threshold

        # La baja saturacion ayuda a priorizar luces blancas sobre colores vivos.
        low_saturation = saturation <= np.percentile(saturation, 70)
        neutral_bright_mask = bright_mask & low_saturation
        if np.count_nonzero(neutral_bright_mask) >= max(1, value.size // 100):
            # Solo se aplica el filtro si conserva suficientes pixeles utiles.
            bright_mask = neutral_bright_mask

        return bright_mask
   
    #para verificar si funciona bien la detección
    def _estimate_direction(self, luminance, bright_mask):
        #Calcula un vector 2D desde el centro hacia la zona mas luminosa."""
        height, width = luminance.shape
        y_coordinates, x_coordinates = np.indices(luminance.shape)
        #La luminancia da mas peso a los puntos realmente intensos.
        weights = luminance * bright_mask
        total_weight = weights.sum()

        if total_weight > 0:
            center_x = float((x_coordinates * weights).sum() / total_weight)
            center_y = float((y_coordinates * weights).sum() / total_weight)
        else:
            #Si no hay peso, se asume una luz centrada y el vector queda en cero.
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0

        # Aqui el vector apunta desde el centro de la imagen hacia la zona brillante.
        direction = np.array([
            center_x - (width - 1) / 2.0,
            center_y - (height - 1) / 2.0,
        ], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            # La normalizacion permite comparar direcciones sin depender del tamano.
            direction /= norm
        return self._to_list(direction)

    def _to_list(self, values):
        #Convierte valores de NumPy a tipos nativos faciles de serializar.
        return [float(value) for value in values]


if __name__ == "__main__":
    image = cv2.imread("data/background/sunset.jpg")
    print(LightingEstimator().estimate(image))