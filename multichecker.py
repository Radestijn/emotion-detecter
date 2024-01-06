import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import time
import random
import os
import requests
from deepface import DeepFace

DATABASE_FOLDER = 'A folder with images of people you want to recognize'

def show_image(image):
    global panel
    image = cv2.resize(image, (250, 250))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if panel.winfo_exists():
        panel.config(image=image)
        panel.image = image
    else:
        panel = tk.Label(image=image)
        panel.image = image
        panel.pack()

def send_to_discord(webhook_url, message):
    data = {
        "content": message,
        "username": "Photo Analysis Bot"
    }
    response = requests.post(webhook_url, json=data)
    return response

def is_person_in_database(frame):
    for filename in os.listdir(DATABASE_FOLDER):
        filepath = os.path.join(DATABASE_FOLDER, filename)
        result = DeepFace.verify(filepath, frame, model_name='VGG-Face', distance_metric='cosine')
        if result["verified"]:
            return True, filename.split(".")[0]
    return False, None

def analyze_photo():
    global panel
    text_area_gender.delete('1.0', tk.END)
    text_area_emotions.delete('1.0', tk.END)
    text_area_combined.delete('1.0', tk.END)

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        text_area_combined.insert(tk.INSERT, "Error: unable to take photo.\n")
        return

    show_image(frame)

    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        analysis = analysis[0]

        emotion = analysis["dominant_emotion"]
        age = analysis["age"]
        gender_probabilities = analysis["gender"]
        man_probability = round(gender_probabilities["Man"], 2)

        if man_probability >= 75.0:
            gender = "Man"
        else:
            gender = "Woman"

        emotions_result = ""
        for emotion_name, emotion_percentage in analysis['emotion'].items():
            emotions_result += f"{emotion_name.capitalize()}: {round(emotion_percentage, 2)}%\n"

        text_area_gender.insert(tk.INSERT, f"Dominant gender: {gender}\n")
        text_area_emotions.insert(tk.INSERT, f"Emotions:\n{emotions_result}")
        text_area_combined.insert(tk.INSERT, f"Dominant emotion: {emotion}\nAge: {age}\n")

        discord_message = f"Dominant Gender: {gender}\nAge: {age}\nDominant Emotion: {emotion}\n\nAll Emotions:\n{emotions_result}"
        send_to_discord('Put here you discord webhook url', discord_message)

        person_found, person_name = is_person_in_database(frame)
        if person_found:
            messagebox.showinfo("Persoon Herkend", f"Persoon herkend als: {person_name}")
        else:
            messagebox.showinfo("Nieuwe Persoon", "Nieuwe persoon gedetecteerd. Gegevens worden opgeslagen.")

    except Exception as e:
        error_message = str(e)
        if "Face could not be detected" in error_message:
            messagebox.showinfo("Geen Persoon Gevonden", "Er is geen persoon gevonden op deze afbeelding")
        else:
            text_area_combined.insert(tk.INSERT, f"Error during face recognition: {error_message}\n")

def start_analysis():
    analyze_photo()

root = tk.Tk()
root.title("Photo Analysis")

frame = tk.Frame(root)
frame.pack()

panel = tk.Label(frame)
panel.pack()

start_button = tk.Button(frame, text="Start Analysis", command=start_analysis)
start_button.pack()

text_area_gender = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=10, width=50)
text_area_gender.pack()

text_area_emotions = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=10, width=50)
text_area_emotions.pack()

text_area_combined = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=10, width=50)
text_area_combined.pack()

root.mainloop()
