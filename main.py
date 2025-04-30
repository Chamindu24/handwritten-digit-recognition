import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageEnhance,ImageChops
import tkinter as tk
from tkinter import filedialog, messagebox

# === Constants ===
MODEL_PATH = 'handwritten_digits_cnn.keras'
IMAGE_DIR = 'digits'
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 15
AUGMENTATION_FACTOR = 5  # How many augmented versions to create per image

# === Utility Functions ===

def load_and_preprocess_data():
    """Load MNIST data with additional custom data augmentation"""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    
    # Data augmentation
    X_train_augmented, y_train_augmented = [], []
    for img, label in zip(X_train, y_train):
        X_train_augmented.append(img)
        y_train_augmented.append(label)
        
        # Create augmented versions
        for _ in range(AUGMENTATION_FACTOR):
            augmented = apply_random_augmentation(img)
            X_train_augmented.append(augmented)
            y_train_augmented.append(label)
    
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_augmented, y_train_augmented, test_size=0.2, random_state=42
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def apply_random_augmentation(img):
    """Apply random augmentation to a single image"""
    # Convert to PIL Image for augmentation
    img_pil = Image.fromarray((img.squeeze() * 255).astype('uint8'))
    
    # Random rotation (-15 to 15 degrees)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR)
    
    # Random zoom (0.9 to 1.1)
    if np.random.rand() > 0.5:
        zoom = np.random.uniform(0.9, 1.1)
        w, h = img_pil.size
        img_pil = img_pil.resize((int(w * zoom), int(h * zoom)), Image.BILINEAR)
        img_pil = ImageOps.fit(img_pil, (w, h), centering=(0.5, 0.5))
    
    # Random brightness/contrast adjustment
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(np.random.uniform(0.8, 1.2))
    
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(np.random.uniform(0.8, 1.2))
    
    # Random shift
    if np.random.rand() > 0.5:
        dx, dy = np.random.randint(-2, 3, size=2)
        img_pil = ImageChops.offset(img_pil, dx, dy)
    
    # Convert back to numpy array
    img_aug = np.array(img_pil).astype('float32') / 255.0
    return img_aug.reshape(IMG_SIZE, IMG_SIZE, 1)

def build_advanced_model():
    """Build a CNN model with dropout and batch normalization"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the model with early stopping and model checkpointing"""
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    model = build_advanced_model()
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'
    )
    
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def load_model():
    """Load the saved model or train a new one if not found"""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No saved model found. Training a new one.")
        return train_model()

def preprocess_custom_image(image_path):
    """Preprocess a custom image for prediction"""
    try:
        # Read and convert to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be read.")
        
        # Threshold and invert (assuming white digit on dark background)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Resize and normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        
        # Center the digit
        img = center_digit(img)
        
        return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def center_digit(img):
    """Center the digit in the image"""
    # Calculate center of mass
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        return img
    
    cX = moments["m10"] / moments["m00"]
    cY = moments["m01"] / moments["m00"]
    
    # Calculate shift needed to center
    shiftX = (IMG_SIZE // 2) - cX
    shiftY = (IMG_SIZE // 2) - cY
    
    # Create transformation matrix
    M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
    centered = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    
    return centered

def predict_image(model, image_path):
    """Make prediction on a single image"""
    img = preprocess_custom_image(image_path)
    if img is None:
        return None, None
    
    prediction = model.predict(img, verbose=0)[0]
    predicted_digit = np.argmax(prediction)
    confidence = prediction[predicted_digit] * 100
    
    # Display results
    plt.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Predicted: {predicted_digit} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()
    
    return predicted_digit, confidence

def predict_custom_images(model):
    """Predict all images in the digits directory"""
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory '{IMAGE_DIR}' not found.")
        return
    
    image_files = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    
    if not image_files:
        print(f"No valid images found in '{IMAGE_DIR}'.")
        return
    
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        digit, confidence = predict_image(model, img_path)
        if digit is not None:
            print(f"[{img_file}] => Predicted: {digit} (Confidence: {confidence:.2f}%)")

def create_gui(model):
    """Create a simple GUI for interaction"""
    root = tk.Tk()
    root.title("Digit Recognition System")
    root.geometry("400x300")
    
    def open_file():
        file_path = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            digit, confidence = predict_image(model, file_path)
            if digit is not None:
                messagebox.showinfo(
                    "Prediction Result",
                    f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%"
                )
    
    # GUI Elements
    tk.Label(root, text="Digit Recognition System", font=('Arial', 16)).pack(pady=10)
    
    tk.Button(
        root, text="Select Image", command=open_file,
        height=2, width=20, font=('Arial', 12)
    ).pack(pady=20)
    
    tk.Button(
        root, text="Predict All in 'digits' Folder", command=lambda: predict_custom_images(model),
        height=2, width=25, font=('Arial', 12)
    ).pack(pady=10)
    
    tk.Button(
        root, text="Retrain Model", command=lambda: [root.destroy(), main()],
        height=1, width=15, font=('Arial', 10)
    ).pack(pady=10)
    
    root.mainloop()

# === Main Function ===

def main():
    print("🧠 Advanced Handwritten Digit Recognition System")
    print("=============================================")
    
    # Check if user wants to retrain
    if os.path.exists(MODEL_PATH):
        choice = input("Found existing model. Retrain? (y/n): ").strip().lower()
        model = train_model() if choice == 'y' else load_model()
    else:
        model = train_model()
    
    # Launch GUI for correction
    create_gui(model)

if __name__ == '__main__':
    main()