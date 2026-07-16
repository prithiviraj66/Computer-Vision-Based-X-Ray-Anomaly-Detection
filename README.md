# Computer Vision-Based X-Ray Anomaly Detection

An unsupervised anomaly detection system built to identify fractures and anomalies in X-ray images. The project leverages a PyTorch-based **Self-Attention Generative Adversarial Network (SAGAN)** to learn the distribution of normal (non-fractured) bone structures, and uses reconstruction error heatmaps to localize anomalies such as bone fractures.

A user-friendly Flask dashboard is included to allow interactive image uploads, DICOM decoding, real-time heatmap visualization, anomaly scoring, and clinical recommendation display.

---

##  Key Features

* **Multi-Format Image Support:** Decodes standard formats (PNG, JPEG, TIFF, BMP, WebP) as well as medical **DICOM** (`.dcm`) images.
* **Pre-processing Pipeline:** Standardized processing matching model training conditions:
  1. Grayscale Conversion
  2. Histogram Equalization (for contrast enhancement)
  3. Aspect-ratio preserving Resize (128x128)
  4. Center Padding
  5. MinMax Normalization
     
* **Unsupervised Reconstruction:** Employs a trained **SAGAN** model to reconstruct input X-rays. Deviations between input and reconstructed structures indicate anomalies (reconstruction loss).
* **Clinical Intelligence Dashboard:**
  * Displays Side-by-Side comparison of original scans and jet-colored anomaly heatmaps.
  * Calculates an Anomaly Score (calibrated MSE normalized to 0-100).
  * Automatically detects body parts (hand/wrist, leg/knee/ankle, hip/pelvis, shoulder/clavicle) to deliver specific emergency and first-aid recommendations.
  * Overlays ground truth Pascal VOC annotation circles for validation.
* **DVC Integration:** Tracks large medical datasets efficiently using Data Version Control (DVC).

---

##  Project Structure

```text
├── app.py                # Main Flask application and REST API server
├── src/                  # Core module packages
│   ├── data/             # Custom datasets and image transforms
│   ├── features/         # Edge detection, grabcut cropping, heatmaps
│   └── models/           # PyTorch definitions (SAGAN, AlphaGAN, Autoencoders)
├── models/               # Model weights storage directory (e.g. SAGAN.pth)
├── notebooks/            # Jupyter notebooks for experiments and visualizations
├── templates/            # Flask HTML templates
├── static/               # Flask CSS and JS dashboard files
├── requirements.txt      # Project dependencies
└── setup.py              # Package installation script
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/prithiviraj66/X-ray-Anomaly-Detection.git
cd X-ray-Anomaly-Detection
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# Create environment
python -m venv venv

# Activate on Windows (Command Prompt)
venv\Scripts\activate
# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Setup Model Weights
Ensure your trained model weights file `SAGAN.pth` is placed inside the `models/` directory:
```bash
models/SAGAN.pth
```
*(Note: If no model file is found, the server automatically starts in a simulated demo fallback mode).*

---

##  Running the Web Application

To run the Flask dashboard locally:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://localhost:5000
```

---

##  How It Works

1. **Reconstruction Principle:** The generator inside the SAGAN is trained to reconstruct normal (non-anomalous) X-rays. 
2. **Reconstruction Error:** When a fractured X-ray is fed into the model, the model fails to reconstruct the fracture itself (as it has only learned to generate normal bone structure).
3. **Anomalous Heatmap:** The absolute difference between the original and reconstructed tensor forms a pixel-wise error map. High reconstruction error points correspond to the anomaly (fracture), visualized on the front-end as a red hot-spot.

## Screenshot
<img width="1885" height="975" alt="Screenshot 2026-07-16 201152" src="https://github.com/user-attachments/assets/b85eaed1-f062-4518-a78b-7947c4ea8e2e" />
<img width="1908" height="888" alt="Screenshot 2026-07-16 201203" src="https://github.com/user-attachments/assets/87ce9653-7497-475b-9ea4-0061afebefe6" />
<img width="1918" height="1077" alt="Screenshot 2026-07-16 201303" src="https://github.com/user-attachments/assets/c49e3c9c-4ee0-4cf3-9a7e-75411f21f4b8" />


