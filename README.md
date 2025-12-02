Hand Tracking Assignment

This project implements real-time hand detection and tracking using Python, OpenCV, and MediaPipe. 
The system detects both left and right hands, identifies 21 key landmarks, and visually tracks hand movement through a webcam feed.

🚀 Features

Real-time webcam-based hand tracking

Detects multiple hands

Extracts 21 hand landmarks per hand

High accuracy using MediaPipe Hands

FPS counter for performance monitoring

Simple, clean code suitable for assignments & learning

Fully extendable for gesture control projects

📸 Demo (What It Does)

Opens webcam

Detects hand(s)

Draws:

Keypoints (21 points)

Hand connections (skeleton)

Displays the result in real-time

🧱 Project Structure
hand-tracking-assignment/
│
├── hand_tracking.py        # Main script
├── requirements.txt        # Required Python libraries
└── README.md               # Documentation

🛠️ Installation
1️⃣ Clone or download the repository
git clone https://github.com/your-username/hand-tracking-assignment.git
cd hand-tracking-assignment

2️⃣ Create and activate virtual environment (recommended)

Windows:

python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Run

Simply run the main script:

python hand_tracking.py


Your webcam will open. Move your hand in front of the camera to see the tracking live.

📦 Requirements

The requirements.txt includes all dependencies, mainly:

Python 3.8+

opencv-python

mediapipe

numpy

🧠 How It Works (Short Explanation)

MediaPipe Hands detects the hand region using machine-learning models.

It computes 21 landmarks for every detected hand.

OpenCV draws the landmarks & connections on each frame.

The script continuously reads frames from the webcam for real-time tracking.

✨ Possible Extensions (For Better Marks)

You can extend this project with:

Finger counting

Hand gesture recognition

Virtual mouse (hand-controlled cursor)

Air drawing (write in the air using hand movements)

Volume control using hand gestures

I can generate these versions too if needed.

👨‍💻 Author    
syed imthiyaz
B.Tech CSE
Viswam Engineering College / JNTUA
