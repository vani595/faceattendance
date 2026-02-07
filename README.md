#  AI Smart Face Attendance System (Anti-Spoofing)

A professional Computer Vision project that uses **Face Recognition** with **Blink Detection** (Liveness Detection) to prevent photo-spoofing and an **Intruder Alert** system to capture unauthorized access.

##  Project Demo
Check out the working demo of the project here:
  **[Watch FaceAttendance Demo Video](https://drive.google.com/file/d/1nJmNJxjo5VDztKnLY8tpSFnKNFJUQxMx/view?usp=drive_link)** 
  

---

##  Key Features
- **Real-time Face Recognition:** Accurately identifies registered individuals.
- **Anti-Spoofing (Blink Detection):** Attendance is only marked when a real human blink is detected (prevents photo-spoofing).
- **Intruder Alert System:** Automatically captures and saves photos of "UNKNOWN" individuals in the `intruders/` folder.
- **Automated Logging:** Saves attendance records with names and timestamps in `Attendance.csv`.

##  Tech Stack
- **Language:** Python 3.12
- **Computer Vision:** OpenCV, Mediapipe
- **Deep Learning:** Face-Recognition (dlib based)
- **Data Management:** NumPy, CSV

##   Project Structure
```text
├── images/               # Put photos of authorized people here
├── intruders/            # Photos of unauthorized people are saved here
├── main.py               # Main application code
├── requirements.txt      # List of dependencies
├── Attendance.csv        # Final attendance log
└── README.md             # Project documentation
