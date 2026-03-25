import os
import tensorflow as tf
from flask import Flask, render_template, request
from utils import preprocess_audio

app = Flask(__name__)

# Path to your saved model (.keras or .h5)
MODEL_PATH = "model/voice_model.keras"   # update if using .h5

# Load the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    if request.method == "POST":
        file = request.files.get("audio")
        if file:
            filepath = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)

            # Preprocess and predict
            X = preprocess_audio(filepath)
            if X is None:
                error_message = "Could not process audio file. Please upload a valid .wav, .mp3, .m4a, or .mp4 file."
            else:
                print(f"DEBUG: Input shape to model = {X.shape}")  # helpful for debugging
                y_pred = (model.predict(X) > 0.5).astype("int32")[0][0]
                prediction = "REAL (Human Voice)" if y_pred == 0 else "FAKE (AI Voice)"
        else:
            error_message = "No audio file uploaded."

    return render_template("index.html", prediction=prediction, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)