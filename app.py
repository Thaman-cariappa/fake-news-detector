from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import os, pickle, re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- GLOBAL MODEL ----------
model = None

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

# ---------- MODEL ----------
def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def train_model(path):
    global model

    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    df["text"] = df["text"].fillna("")
    X = df["text"].apply(clean_text)
    y = df["label"].str.upper()

    model = build_model()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

# Load model at start
load_model()

# ---------- FULL HTML ----------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Fake News Detector</title>
<style>

body {
    margin:0;
    font-family: 'Segoe UI';
    background: linear-gradient(135deg,#141e30,#243b55);
    color:white;
    text-align:center;
}

/* HEADER */
h1 {
    margin-top:40px;
    font-size:48px;
    animation: fadeIn 2s ease;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:25px;
    margin:30px auto;
    width:55%;
    border-radius:20px;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
    animation: slideUp 1s ease;
}

/* UPLOAD */
.upload-box {
    border:2px dashed #aaa;
    padding:40px;
    border-radius:15px;
    cursor:pointer;
}
.upload-box:hover {
    border-color:#00c6ff;
    background:rgba(0,198,255,0.1);
}

/* INPUT */
textarea {
    width:80%;
    height:120px;
    border:none;
    border-radius:10px;
    padding:10px;
}

/* BUTTON */
button {
    padding:12px 25px;
    border:none;
    border-radius:10px;
    background:#00c6ff;
    cursor:pointer;
}
button:hover {
    background:#0072ff;
    color:white;
}

/* PROGRESS */
.progress {
    width:80%;
    margin:10px auto;
    background:#333;
    border-radius:10px;
}
.progress-bar {
    height:10px;
    width:0%;
    background:#00c6ff;
}

/* RESULT */
.result-box {
    margin-top:15px;
    font-size:20px;
    font-weight:bold;
}

/* ANIMATION */
@keyframes fadeIn {
    from{opacity:0;}
    to{opacity:1;}
}
@keyframes slideUp {
    from{transform:translateY(50px);opacity:0;}
    to{transform:translateY(0);opacity:1;}
}

</style>
</head>

<body>

<h1>🧠 AI Fake News Detector</h1>

<!-- TRAIN -->
<div class="card">
<h2>Upload Dataset</h2>
<form id="trainForm">
    <div class="upload-box">
        <input type="file" name="file" required>
        <p>Upload CSV</p>
    </div>
    <button type="submit">Train Model</button>
</form>

<div class="progress"><div class="progress-bar" id="trainBar"></div></div>
<p id="trainStatus"></p>
</div>

<!-- PREDICT -->
<div class="card">
<h2>Analyze News</h2>
<form id="predictForm">
    <textarea name="text" placeholder="Paste news text..." required></textarea>
    <br><br>
    <button type="submit">Predict</button>
</form>

<div class="progress"><div class="progress-bar" id="predBar"></div></div>
<div class="result-box" id="result"></div>
</div>

<script>

// TRAIN
document.getElementById("trainForm").onsubmit = async (e)=>{
    e.preventDefault();
    let bar = document.getElementById("trainBar");
    bar.style.width="50%";

    let formData = new FormData(e.target);

    let res = await fetch("/train",{method:"POST",body:formData});
    let data = await res.json();

    bar.style.width="100%";
    document.getElementById("trainStatus").innerText=data.status;
};

// PREDICT
document.getElementById("predictForm").onsubmit = async (e)=>{
    e.preventDefault();
    let bar = document.getElementById("predBar");
    bar.style.width="60%";

    let formData = new FormData(e.target);

    let res = await fetch("/predict",{method:"POST",body:formData});
    let data = await res.json();

    bar.style.width="100%";

    if(data.error){
        document.getElementById("result").innerText=data.error;
    } else {
        let color = data.prediction==="FAKE" ? "red":"lightgreen";
        document.getElementById("result").innerHTML =
            "<span style='color:"+color+"'>" +
            data.prediction + " ("+data.confidence+"%)</span>";
    }
};

</script>

</body>
</html>
"""

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "No file uploaded ❌"})

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        train_model(path)

        return jsonify({"status": "Model trained successfully 🚀"})

    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Train model first!"})

        text = request.form.get("text", "")
        if not text.strip():
            return jsonify({"error": "Enter some text!"})

        cleaned = clean_text(text)

        pred = model.predict([cleaned])[0]
        prob = max(model.predict_proba([cleaned])[0]) * 100

        return jsonify({
            "prediction": pred,
            "confidence": round(prob, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)