import cv2
from processing.lighting import LightingEstimator


def run_realtime_lighting_demo(device_id=0):
    #Aca mostramos en vivo la estimacion de luz de la webcam
    estimator = LightingEstimator()
    camera = cv2.VideoCapture(device_id)

    while True:
        ok, frame = camera.read()

        # Cada frame se analiza por separado, por eso los valores pueden cambiar.
        # La estimacion se actualiza para cada frame de la webcam.
        lighting = estimator.estimate(frame)
        direction = lighting["direction"]
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        # Se escala el vector para que pueda verse como una flecha sobre el frame.
        arrow_end = (
            int(center[0] + direction[0] * width * 0.25),
            int(center[1] + direction[1] * height * 0.25),
        )
        # La flecha permite comprobar visualmente la direccion detectada.
        cv2.arrowedLine(frame, center, arrow_end, (0, 255, 255), 3, tipLength=0.2)

        overlay_lines = [
            f"Direccion: ({direction[0]:.2f}, {direction[1]:.2f})",
            f"Color RGB: {[round(value, 2) for value in lighting['color']]}",
            f"Intensidad: {lighting['intensity']:.2f}",
            "Q: salir",
        ]
        for index, text in enumerate(overlay_lines):
            # Los parametros numericos se dibujan para validar el resultado.
            #Ignora esto, solo era para debuggear y ver los valores
            cv2.putText(
                frame,
                text,
                (10, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Realtime Lighting Detector", frame)
    # Solo es la configuración para cerrar la camara (Usa Q).
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_lighting_demo()