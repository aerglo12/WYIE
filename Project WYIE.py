import cv2
import numpy as np

cap = cv2.VideoCapture(0)

previous_color = None
THRESHOLD = 40  # makin kecil makin sensitif

def get_average_color(frame):
    h, w, _ = frame.shape
    center = frame[h//2-50:h//2+50, w//2-50:w//2+50]
    avg_color = np.mean(center, axis=(0,1))
    return avg_color

def color_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    avg_color = get_average_color(frame)

    if previous_color is not None:
        delta = color_distance(avg_color, previous_color)

        if delta > THRESHOLD:
            print(f"COLOR TRANSFORM DETECTED! Δ = {delta:.2f}")

            cv2.putText(frame, "COLOR CHANGED!",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

    previous_color = avg_color

    # Kotak tengah
    h, w, _ = frame.shape
    cv2.rectangle(frame,
                  (w//2-50, h//2-50),
                  (w//2+50, h//2+50),
                  (0,255,0),
                  2)

    cv2.imshow("Sensitive Color Transform Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
