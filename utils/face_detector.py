import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import numpy as np

class FaceDetector:
    def __init__(self):
        pass

    def detect_faces(self, image_path: str):
        faces = RetinaFace.detect_faces(image_path)
        return faces

    def face_features(self, faces):
        features = []
        for face in faces.values():
            landmarks = face['landmarks']
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            mouth_left = landmarks.get('mouth_left')
            mouth_right = landmarks.get('mouth_right')
            bounding_box = face['facial_area']
            features.append((left_eye, right_eye, nose, mouth_left, mouth_right, bounding_box))
        return features

    def is_frontface(self, features, eye_threshold=30, nose_threshold=35):
        for f in features:
            left_eye, right_eye, nose, mouth_left, mouth_right, _ = f
            
            # Calculate the symmetry of the face
            eye_nose_diff = abs(left_eye[0] + right_eye[0]) / 2 - nose[0]
            mouth_nose_diff = abs(mouth_left[0] + mouth_right[0]) / 2 - nose[0]
            
            # Calculate distances
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            nose_to_eye_mid = abs((left_eye[1] + right_eye[1]) / 2 - nose[1])
            
            calculations = {
                'eye_distance': eye_distance,
                'nose_to_eye_mid': nose_to_eye_mid,
                'eye_nose_diff': eye_nose_diff,
                'mouth_nose_diff': mouth_nose_diff,
            }

            conditions = {
                'eye_distance': eye_distance > eye_threshold,
                'nose_to_eye_mid': nose_to_eye_mid < nose_threshold,
                'eye_nose_diff': eye_nose_diff < eye_threshold,
                'mouth_nose_diff': mouth_nose_diff < eye_threshold
            }
            
            is_front = all(conditions.values())
            
            return is_front, calculations, conditions 

    def visualize_detection(self, image_path, features):
        # Load the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for feature in features:
            left_eye, right_eye, nose, mouth_left, mouth_right, bbox = feature

            cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            cv2.circle(img_rgb, tuple(map(int, left_eye)), 5, (0, 255, 0), -1)
            cv2.circle(img_rgb, tuple(map(int, right_eye)), 5, (0, 255, 0), -1)
            cv2.circle(img_rgb, tuple(map(int, nose)), 5, (0, 255, 0), -1)
            cv2.circle(img_rgb, tuple(map(int, mouth_left)), 5, (0, 255, 0), -1)
            cv2.circle(img_rgb, tuple(map(int, mouth_right)), 5, (0, 255, 0), -1)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()