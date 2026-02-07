import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mediapipe as mp
import sys
import os

# Ye line Python ko batayegi ki models yahan chhupay hain
models_path = r'C:\Users\vanit\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages'
sys.path.append(models_path)

# --- 1. Mediapipe Setup (Sabse Simple aur Direct Tarika) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Aankhon ke points
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Global variable for blink
blink_detected = False

# --- 2. Images load karne ka setup ---
path = 'images'
images = []
classNames = []

if not os.path.exists(path):
    os.makedirs(path)

myList = os.listdir(path)
print(f'Logon ki list mili: {myList}')

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(f'Attendance Marked: {name}')

def get_aspect_ratio(landmarks, eye_points):
    p1 = landmarks[eye_points[1]]
    p5 = landmarks[eye_points[5]]
    p2 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[4]]
    dist = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p5.x, p5.y]))
    dist2 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p4.x, p4.y]))
    return (dist + dist2) / 2

# Check if images exist
if len(images) > 0:
    encodeListKnown = findEncodings(images)
    print('Encoding Done! Camera Start Ho Raha Hai...')
else:
    print('Error: "images" folder mein koi photo nahi mili!')
    exit()

# --- 3. Camera Start ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Blink Detection
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        mesh_coords = results.multi_face_landmarks[0].landmark
        ratio = get_aspect_ratio(mesh_coords, LEFT_EYE)
        
        # Blink detect logic (Threshold 0.015)
        if ratio < 0.018: 
            blink_detected = True

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if matches[matchIndex] and faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            
            if blink_detected:
                color = (0, 255, 0) # Green
                markAttendance(name)
                status = "Status: Verified & Marked"
            else:
                color = (0, 165, 255) # Orange
                status = "Status: Please Blink"
        else:
            name = "UNKNOWN"
            color = (0, 0, 255) # Red
            status = "Status: Unauthorized"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Attendance System (Anti-Spoofing)', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()