import cv2
import numpy as np

def process_video():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize background video/image
    background = cv2.imread('data/background/sunset.jpg')
    if background is None:
        background = np.zeros((480, 640, 3), dtype=np.uint8)
    background = cv2.resize(background, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up mask with morphological closing (rellena huecos sin generar bloques)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # Kernel más pequeño para el open: solo quita ruido de 1-2 píxeles sueltos,
        # sin borrar franjas delgadas de verde detectado entre los dedos
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # Suavizar bordes: pasa la máscara binaria a un degradado 0-255
        # en vez de un corte duro, para eliminar el efecto "dientes de sierra"
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Normalizar a alpha [0,1] float, con 3 canales para poder multiplicar la imagen a color
        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        # Blending: mezcla proporcional en vez de "todo o nada"
        fg = frame.astype(np.float32)
        bg = background.astype(np.float32)
        result = fg * (1 - alpha) + bg * alpha
        result = result.astype(np.uint8)
        
        # Display result
        cv2.imshow('Green Screen', result)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()