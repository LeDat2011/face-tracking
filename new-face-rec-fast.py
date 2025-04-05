import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import time

# Khởi tạo Firebase
cred = credentials.Certificate(r"D:\SMARTHOME\FACE-TRACKING\FACE-MAIN\testesp32-7e391-firebase-adminsdk-xo01w-69f96d0823.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://testesp32-7e391-default-rtdb.firebaseio.com/'
})

# Đường dẫn đến thư mục chứa khuôn mặt đã biết
known_faces_dir = "known_faces"
unknown_faces_dir = "unknown_faces"
output_dir = "output_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Tải mã hóa và tên khuôn mặt đã biết
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        except IndexError:
            print(f"Warning: No face found in {filename}. Skipping this file.")

# Hàm chuyển đổi trạng thái cửa trên Firebase
def toggle_door_status():
    ref = db.reference("field/doorStatus")
    current_status = ref.get()
    new_status = "CLOSE" if current_status == "OPEN" else "OPEN"
    ref.set(new_status)
    print(f"Door status toggled: {new_status}")

# Khởi tạo webcam
video_capture = cv2.VideoCapture(1)

print("Starting video stream. Press 'q' to quit.")

# Khởi tạo các biến
face_locations = []
face_encodings = []
process_this_frame = True
last_detected_time = None
stable_face_location = None
stable_face_time = 2  # Thời gian yêu cầu khuôn mặt giữ nguyên (giây)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:
            current_time = time.time()

            # Kiểm tra nếu phát hiện khuôn mặt giống vị trí cũ
            if stable_face_location and face_locations[0] == stable_face_location:
                # Nếu cùng vị trí, kiểm tra thời gian
                if current_time - last_detected_time >= stable_face_time:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    print("Face stabilized for 2 seconds. Capturing image...")

                    full_frame_path = os.path.join(unknown_faces_dir, f"unknown_full_frame_{timestamp}.jpg")
                    cv2.imwrite(full_frame_path, frame)

                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        face_image = frame[top:bottom, left:right]
                        unknown_face_locations = [(top, right, bottom, left)]
                        unknown_face_encoding = face_recognition.face_encodings(frame, unknown_face_locations)[0]

                        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.5:
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                toggle_door_status()

                        cropped_face = frame[top:bottom, left:right].copy()
                        cv2.rectangle(cropped_face, (0, 0), (right-left, bottom-top), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(cropped_face, name, (6, bottom-top-6), font, 0.5, (255, 255, 255), 1)
                        output_path = os.path.join(output_dir, f"{name}_{timestamp}_{i}.jpg")
                        cv2.imwrite(output_path, cropped_face)

                    stable_face_location = None
            else:
                # Cập nhật vị trí khuôn mặt và thời gian phát hiện
                stable_face_location = face_locations[0]
                last_detected_time = current_time

    process_this_frame = not process_this_frame

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
