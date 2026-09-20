import cv2
from processing.lighting import LightingEstimator


def run_realtime_lighting_demo(device_id=0):
    estimator = LightingEstimator()
    camera = cv2.VideoCapture(device_id)

    while True:
        ok, frame = camera.read()

        lighting = estimator.estimate(frame)
        direction = lighting["direction"]
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        arrow_end = (
            int(center[0] + direction[0] * width * 0.25),
            int(center[1] + direction[1] * height * 0.25),
        )
        cv2.arrowedLine(frame, center, arrow_end, (0, 255, 255), 3, tipLength=0.2)

        overlay_lines = [
            f"Direccion: ({direction[0]:.2f}, {direction[1]:.2f})",
            f"Color RGB: {[round(value, 2) for value in lighting['color']]}",
            f"Intensidad: {lighting['intensity']:.2f}",
            "Q: salir",
        ]
        for index, text in enumerate(overlay_lines):
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_lighting_demo()