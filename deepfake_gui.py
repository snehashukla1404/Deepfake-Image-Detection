import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import ttk 
from PIL import Image, ImageTk
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model


model = load_model("deepfake_detector_3.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)) / 255.0  
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)[0][0]
    return "Fake" if prediction > 0.5 else "Real"

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        result_text.set("Processing...")
        progress_bar.start(10)  

        root.after(1000, lambda: process_prediction(file_path))  

def process_prediction(file_path):
    result = predict_image(file_path)
    result_text.set(f"Prediction: {result}")
    progress_bar.stop()  

    
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img


root = tk.Tk()
root.title("Deepfake Detector")
root.geometry("500x600")
root.configure(bg="#2E2E2E")  


style = ttk.Style()
style.configure("TButton",
                font=("Arial", 12, "bold"),
                padding=10,
                background="#007BFF",
                foreground="black")  
style.map("TButton", foreground=[("pressed", "white"), ("active", "white")], background=[("active", "#0056b3")])


upload_btn = ttk.Button(root, text="📂 Upload Image", command=upload_image, style="TButton")
upload_btn.pack(pady=20)


img_label = Label(root, bg="#2E2E2E")
img_label.pack()


progress_bar = ttk.Progressbar(root, mode="indeterminate", length=250)
progress_bar.pack(pady=15)


result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Arial", 14, "bold"), fg="white", bg="#2E2E2E")
result_label.pack(pady=20)

root.mainloop()
