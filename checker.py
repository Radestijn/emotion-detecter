import cv2
from deepface import DeepFace
import time

# Model en cascade classifier laden
model = DeepFace.build_model("Emotion")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Start de camera
cap = cv2.VideoCapture(0)
time.sleep(2)
emotions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fout: Frame niet gelezen.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_roi = gray_frame[y:y + h, x:x + w]

        try:
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]
            emotions[index] = emotion
        except Exception as e:
            print("Fout tijdens emotiedetectie: ", e)
            emotions[index] = "Error"

        cv2.putText(frame, f"Emotie: {emotions[index]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Aantal mensen: {len(faces)}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()