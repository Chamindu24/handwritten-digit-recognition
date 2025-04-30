🧠 Handwritten Digit Recognition
A Python project using a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from images or webcam input. Built with TensorFlow, it achieves ~99% accuracy on the MNIST dataset and includes a Tkinter GUI.
✨ Features

Predicts digits from images or webcam with high accuracy.
Simple GUI for file uploads and live predictions.
Saves predictions to a file.
Visualizes training performance (accuracy/loss, confusion matrix).
Handles noisy images with advanced preprocessing.

🛠️ Requirements

Python 3.7+
Packages: numpy, opencv-python, matplotlib, tensorflow, scikit-learn, seaborn, pillow

🚀 Installation

Clone or download the project:git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition


Set up a virtual environment (optional):python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt



📂 Setup

Place digit images in the digits/ folder (e.g., digit1.png).
Ensure a webcam is connected for live predictions.

🖌️ Usage

Run the script:python digit_recognition.py


Choose to train a new model (y) or load an existing one (n).
Use the GUI:
Select Image: Upload an image.
Use Webcam: Press 'q' to capture, 'esc' to exit.
Save Predictions: Export results to a file.



🎯 Example
🧠 Handwritten Digit Recognition (Upgraded v2.0)
Train a new model? (y/n): n


GUI: "Predicted: 7 (95.23%)"
Output file: predictions_20250430_123456.txt

🤝 Contributing
Fork, make changes, and submit a pull request. Issues and ideas welcome!
📜 License
MIT License. See LICENSE.
📬 Contact
Questions? Email [your-email@example.com] or open a GitHub issue.
Happy digit recognizing! 🚀
