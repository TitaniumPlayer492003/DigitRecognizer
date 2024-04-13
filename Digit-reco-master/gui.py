import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# model = tf.keras.models.load_model('hand.model')
model = tf.keras.models.load_model('hand1.model')


# wind
window = tk.Tk()
window.title("Digit Classification")
window.geometry("300x400")

# canvas to display
canvas = tk.Canvas(window, width=200, height=200, bg="white")
canvas.pack(pady=20)

# Creating a label to display the prediction
label_prediction = tk.Label(window, text="Prediction: ", font=("Helvetica", 16))
label_prediction.pack()


# Function to classify the digit
def classify_digit():
    # Load the digit image from the file
    image_path = entry_image_path.get()
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = img.reshape(1, 28, 28)

        # Predict the digit
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        # Display the digit image on the canvas
        image = Image.fromarray(np.uint8(img[0]))
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=image)
        canvas.image = image

        # Update prediction
        label_prediction.config(text="Prediction: " + str(digit))
    except:
        messagebox.showerror("Error", "Failed to load or process the image.")


#image path
entry_image_path = tk.Entry(window, width=30)
entry_image_path.pack(pady=10)

# classify
button_classify = tk.Button(window, text="Classify", command=classify_digit)
button_classify.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()