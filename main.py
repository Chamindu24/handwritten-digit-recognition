import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog, messagebox
import seaborn as sns
import datetime

# === Constants ===
MODEL_PATH = 'handwritten_digits_cnn.keras'
IMAGE_DIR = 'digits'
IMG_SIZE = 28
EPOCHS = 10
BATCH_SIZE = 32

# === Utility Functions ===

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # Normalize to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test

def build_model():
    """Build a Convolutional Neural Network (CNN)."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Prevent overfitting
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train the model with data augmentation and early stopping."""
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    model = build_model()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    
    return model

def load_model():
    """Load a saved model or train a new one."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No saved model found. Training a new one.")
        return train_model()

def preprocess_image(img):
    """Preprocess a single image."""
    if len(img.shape) == 3:  # Convert color to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Adaptive thresholding to handle varying backgrounds
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.bitwise_not(img)  # Invert (white digits on black)
    img = img / 255.0  # Normalize
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict_image(model, img):
    """Predict digit from a single image."""
    img = preprocess_image(img)
    prediction = model.predict(img, verbose=0)[0]
    predicted_digit = np.argmax(prediction)
    confidence = prediction[predicted_digit] * 100
    return predicted_digit, confidence

def save_predictions(predictions):
    """Save predictions to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'predictions_{timestamp}.txt', 'w') as f:
        for pred in predictions:
            f.write(f"Image: {pred['image']}, Predicted: {pred['digit']}, Confidence: {pred['confidence']:.2f}%\n")
    print(f"Predictions saved to predictions_{timestamp}.txt")

# === GUI Application ===

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.model = load_model()
        self.predictions = []
        
        # GUI elements
        tk.Label(root, text="Handwritten Digit Recognition", font=("Arial", 16)).pack(pady=10)
        
        tk.Button(root, text="Select Image", command=self.predict_from_file).pack(pady=5)
        tk.Button(root, text="Use Webcam", command=self.predict_from_webcam).pack(pady=5)
        tk.Button(root, text="Save Predictions", command=self.save_predictions).pack(pady=5)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=IMG_SIZE*10, height=IMG_SIZE*10)
        self.canvas.pack(pady=10)
        
    def predict_from_file(self):
        """Predict digit from a selected file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            try:
                img = cv2.imread(file_path)
                digit, confidence = predict_image(self.model, img)
                self.predictions.append({'image': os.path.basename(file_path), 'digit': digit, 'confidence': confidence})
                self.result_label.config(text=f"Predicted: {digit} ({confidence:.2f}%)")
                
                # Display image
                img_display = cv2.resize(img, (IMG_SIZE*10, IMG_SIZE*10))
                img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                img_display = tk.PhotoImage(data=cv2.imencode('.png', img_display)[1].tobytes())
                self.canvas.create_image(0, 0, anchor='nw', image=img_display)
                self.canvas.image = img_display
            except Exception as e:
                messagebox.showerror("Error", f"Could not process image: {e}")
    
    def predict_from_webcam(self):
        """Predict digit from webcam capture."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam - Press 'q' to capture, 'esc' to exit", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                try:
                    digit, confidence = predict_image(self.model, frame)
                    self.predictions.append({'image': 'webcam_capture', 'digit': digit, 'confidence': confidence})
                    self.result_label.config(text=f"Predicted: {digit} ({confidence:.2f}%)")
                    
                    # Display captured frame
                    frame_display = cv2.resize(frame, (IMG_SIZE*10, IMG_SIZE*10))
                    frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                    frame_display = tk.PhotoImage(data=cv2.imencode('.png', frame_display)[1].tobytes())
                    self.canvas.create_image(0, 0, anchor='nw', image=frame_display)
                    self.canvas.image = frame_display
                except Exception as e:
                    messagebox.showerror("Error", f"Could not process image: {e}")
            elif key == 27:  # Escape key
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_predictions(self):
        """Save all predictions to a file."""
        if self.predictions:
            save_predictions(self.predictions)
            messagebox.showinfo("Success", "Predictions saved successfully")
        else:
            messagebox.showwarning("Warning", "No predictions to save")

# === Main ===

if __name__ == '__main__':
    print("🧠 Handwritten Digit Recognition (Upgraded v2.0)")
    choice = input("Train a new model? (y/n): ").strip().lower()
    if choice == 'y':
        train_model()
    
    # Start GUI
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()