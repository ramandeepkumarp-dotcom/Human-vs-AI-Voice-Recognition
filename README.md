🎙️ Human vs AI Voice Recognition
This project demonstrates how deep learning can distinguish between human voices and AI‑generated voices. It combines TensorFlow/Keras for model inference, Librosa for audio preprocessing, and Flask for deployment into a modern web application.

📂 Project Structure
voice_flask_app/
│
├── app.py                # Flask application
├── utils.py              # Audio preprocessing functions
├── model/
│   └── voice_model.keras # Trained CNN model
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   └── style.css         # Frontend CSS
└── uploads/              # Uploaded audio files



📖 Project Overview
The Human vs AI Voice Recognition system is designed to detect whether an uploaded audio file contains a real human voice or an AI‑generated synthetic voice.
- Problem Statement: With the rise of generative AI, distinguishing authentic voices from synthetic ones is becoming critical for security, media verification, and fraud prevention.
- Solution: A convolutional neural network (CNN) trained on spectrograms of audio samples, deployed via Flask, and accessible through a clean web interface.
- Outcome: A recruiter‑ready portfolio project showcasing end‑to‑end machine learning deployment skills.

🛠️ Tech Stack
- Python 3.13
- TensorFlow / Keras → Model training & inference
- Librosa → Audio preprocessing (Mel spectrograms)
- Flask → Web framework
- HTML / CSS → Frontend design
- FFmpeg → MP3/M4A decoding support

⚙️ Workflow
- Upload Audio: User selects .wav, .mp3, .m4a, or .mp4 file.
- Preprocessing:
- Audio is loaded with Librosa.
- Converted into a 128×128 Mel spectrogram.
- Normalized and reshaped to (128, 128, 1).
- Prediction:
- CNN model classifies input as REAL (Human Voice) or FAKE (AI Voice).
- Result Display:
- Prediction shown on the web interface with styled success/error messages.

🎨 Web Interface Features
- Navigation bar with Home, About, How it Works, Contact.
- About section explaining project purpose and tech stack.
- How it Works section with step‑by‑step workflow.
- Contact section with developer details.
- Responsive design with gradient background, animations, and styled buttons.

🧠 Model Details
- Input: 128×128 Mel spectrograms reshaped to (128, 128, 1).
- Architecture: Convolutional Neural Network (CNN).
- Output: Binary classification → Human Voice (0) or AI Voice (1).

🚀 Setup Instructions
- Clone the repository:
git clone https://github.com/yourusername/voice-flask-app.git
cd voice-flask-app
- Install dependencies:
pip install -r requirements.txt
- Install FFmpeg:
- Download from ffmpeg.org (ffmpeg.org in Bing).
- Add the bin/ folder to your system PATH.
- Run the Flask app:
python app.py
- Open in browser:
http://127.0.0.1:5000

✨ This project highlights end‑to‑end ML deployment skills: audio preprocessing, model building, and full‑stack integration. It’s designed as a recruiter‑ready portfolio project to showcase both technical and presentation skills.