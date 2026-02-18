import cv2
import numpy as np
from PIL import Image
from hsv_limit_function import get_limits_adv

# TO CAPTURE WEB CAM
# url = "http://10.121.1.109:8080/video"
cap = cv2.VideoCapture(0)

# HSV COLOR RANGES
HSV_COLORS = {
    "red": [
        (0, 120, 70, 10, 255, 255),
        (170, 120, 70, 179, 255, 255)
    ],
    "green": [(35, 80, 80, 85, 255, 255)],
    "blue": [(90, 80, 80, 130, 255, 255)],
    "yellow": [(20, 100, 100, 35, 255, 255)],
    "orange": [(10, 100, 100, 20, 255, 255)],
    "pink": [(140, 50, 50, 170, 255, 255)]
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Convert BGR → HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through each color
    for color_name, ranges in HSV_COLORS.items():

        mask = None

        # Handle single / multiple HSV ranges
        for hsv_range in ranges:
            lower, upper = get_limits_adv(hsv_range)
            current_mask = cv2.inRange(hsv_frame, lower, upper)

            if mask is None:
                mask = current_mask
            else:
                mask = mask | current_mask

        # NumPy → PIL (your method)
        mask_pil = Image.fromarray(mask)
        bbox = mask_pil.getbbox()

        if bbox:
            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{color_name.upper()}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
