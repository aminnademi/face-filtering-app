from retinaface import RetinaFace

class FaceDetector:
    def __init__(self):
        pass

    def detect_faces(self, image_path: str):
        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(image_path)
        return faces

    def face_features(self, faces):
        features = []
        for face in faces.values():
            landmarks = face['landmarks']
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            features.append((left_eye, right_eye, nose))
        return features

    def is_frontface(self, features, eye_threshold=20, nose_threshold=20):
        for i in features:
            left_eye, right_eye, nose = i
            if abs(left_eye[0] - right_eye[0]) < eye_threshold and abs((left_eye[1] + right_eye[1]) / 2 - nose[1]) < nose_threshold:
                return True
        return False
