import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

def crop_nose_to_chin(image, side_margin=0.2, chin_margin=0.1):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected.")

    lm = results.multi_face_landmarks[0].landmark
    nose_y = int(lm[1].y * (h+50))
    chin_y = int(lm[152].y * (h+50))
    left_x = int(lm[234].x * (w+100))
    right_x = int(lm[454].x * (w-100))

    width = right_x - left_x
    left_x = max(0, left_x - int(width * side_margin))
    right_x = min(w, right_x + int(width * side_margin))
    top_y = max(0, int(nose_y - width * 0.1))

    # extend below chin
    bottom_y = min(h, chin_y + int((chin_y - top_y) * chin_margin))

    crop = img[top_y:bottom_y, left_x:right_x]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))