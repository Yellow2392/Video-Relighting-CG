import cv2
import numpy as np

class GreenScreenProcessor:
    def __init__(self, background_path='data/background/sunset.jpg', target_size=(640, 480)):
        self.target_size = target_size
        
        bg = cv2.imread(background_path)
        if bg is None:
            bg = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        self.background = cv2.resize(bg, self.target_size)
        
        self.lower_green = np.array([40, 40, 40]) # rangos HSV
        self.upper_green = np.array([80, 255, 255])
        
        # kernels precalculados
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _process_frame(self, frame):
        # segmentación + alpha blending sobre un frame
        frame_resized = cv2.resize(frame, self.target_size)
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        fg = frame_resized.astype(np.float32)
        bg = self.background.astype(np.float32)
        
        result = fg * (1.0 - alpha) + bg * alpha
        return result.astype(np.uint8)

    def _run_stream(self, cap, delay=1, output_path=None):
        # reproducción y guardado opcional
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, self.target_size)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            result = self._process_frame(frame)

            if writer:
                writer.write(result)

            cv2.imshow('Green Screen', result)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def process_webcam(self, device_id=0): # Tiempo real desde webcam
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {device_id}")
            return
        self._run_stream(cap, delay=1)

    def process_video_file(self, video_path, output_path=None): # Archivo .mp4 existente 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el archivo {video_path}")
            return

        #fps = cap.get(cv2.CAP_PROP_FPS) # ajuste para la velocidad real de reproducción
        #delay = int(1000 / fps) if fps > 0 else 60

        #self._run_stream(cap, delay=delay, output_path=output_path)
        self._run_stream(cap, output_path=output_path)