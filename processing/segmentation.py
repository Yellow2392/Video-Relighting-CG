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
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Invert mask for foreground
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract foreground and background
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        bg = cv2.bitwise_and(background, background, mask=mask)
        
        # Combine foreground and background
        result = cv2.add(fg, bg)
        
        # Display result
        cv2.imshow('Green Screen', result)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()