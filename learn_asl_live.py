# Importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the trained MLP model
model = tf.keras.models.load_model("best_model_combined.keras")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Load example images for each letter 
image_path = "/Users/ahadm/Desktop/Alpha"
letter_images = {
    label: cv2.imread(os.path.join(image_path, f"{label}.png")) for label in labels
}

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize global variables for score tracking
score = 0
skipped = 0
root = None

# Function to restart learning session
def restart_learning():
    global root
    root.destroy()
    main_ui()

# Function to start the learning session
def start_learning():
    global score, skipped, root
    current_index = 0
    score = 0
    skipped = 0
    confirmed = 0
    last_prediction = "---"

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if current_index >= len(labels):
            break

        letter = labels[current_index]
        example_img = letter_images.get(letter, None)

        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction = "---"
        feedback = "Try the sign."

        # Detect hand landmarks and make prediction
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    input_data = np.array(landmarks).reshape(1, -1)
                    predicted_class = np.argmax(model.predict(input_data, verbose=0))
                    if predicted_class < len(labels):
                        predicted_letter = labels[predicted_class]
                        prediction = predicted_letter

                        # Check if prediction matches the current letter
                        if predicted_letter == letter:
                            if last_prediction == predicted_letter:
                                confirmed += 1
                            else:
                                confirmed = 1
                            last_prediction = predicted_letter

                            if confirmed >= 5:
                                feedback = "Correct! Press ENTER to continue."
                                if example_img is not None:
                                    resized_example = cv2.resize(example_img, (200, 200))
                                    frame[10:210, 10:210] = resized_example

                                cv2.putText(frame, f"Target: {letter}", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                cv2.putText(frame, f"Prediction: {prediction}", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
                                cv2.putText(frame, feedback, (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                                cv2.putText(frame, f"Score: {score} | Skipped: {skipped}", (230, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                cv2.imshow("ASL Learning Assistant", frame)

                                # Wait for user confirmation to proceed
                                while True:
                                    key = cv2.waitKey(0) & 0xFF
                                    if key == 13:
                                        score += 1
                                        current_index += 1
                                        confirmed = 0
                                        break
                                    elif key == ord('q'):
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        return
                                continue

        # Display example image and instructions
        if example_img is not None:
            resized_example = cv2.resize(example_img, (200, 200))
            frame[10:210, 10:210] = resized_example

        cv2.putText(frame, f"Target: {letter}", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {prediction}", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        cv2.putText(frame, feedback, (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        cv2.putText(frame, f"Score: {score} | Skipped: {skipped}", (230, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'S' to skip", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("ASL Learning Assistant", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            skipped += 1
            current_index += 1
            confirmed = 0
        elif key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Show result summary in Tkinter window
    result_window = tk.Tk()
    result_window.title("Session Result")
    result_window.geometry("350x200")
    result_window.configure(bg="#222")

    label = tk.Label(result_window, text=f"Your Score: {score}/{len(labels)}\Skipped: {skipped}", font=("Arial", 14), bg="#222", fg="white")
    label.pack(pady=20)

    retry_button = tk.Button(result_window, text="Retry", font=("Arial", 12), command=lambda: [result_window.destroy(), restart_learning()])
    retry_button.pack(pady=5)

    exit_button = tk.Button(result_window, text="Exit", font=("Arial", 12), command=result_window.destroy)
    exit_button.pack(pady=5)

    result_window.mainloop()

# Main UI setup using Tkinter
def main_ui():
    global root
    root = tk.Tk()
    root.title("ASL Learning")
    root.geometry("400x400")
    root.configure(bg="#1e1e1e")

    try:
        logo_img = Image.open("ASL_Alphabet.jpg")
        logo_img = logo_img.resize((250, 250))
        logo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(root, image=logo, bg="#1e1e1e")
        logo_label.image = logo
        logo_label.pack(pady=10)
    except:
        pass

    label = tk.Label(root, text="Do you want to learn American Sign Language ( Only Letters )?", font=("Arial", 12), bg="#1e1e1e", fg="white")
    label.pack(pady=10)

    start_button = tk.Button(root, text="Start Learning", font=("Arial", 12), command=lambda: [root.destroy(), start_learning()])
    start_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit)
    exit_button.pack(pady=5)

    root.mainloop()
# Start the application
main_ui()