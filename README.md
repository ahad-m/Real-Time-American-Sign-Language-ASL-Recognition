# Real-Time-American-Sign-Language-ASL-Recognition
This project develops a real-time system for recognizing American Sign Language (ASL) hand signs (A-Z) using a Multi-Layer Perceptron (MLP) model and MediaPipe for hand landmark extraction. The system aims to provide an efficient and accurate solution for ASL recognition, suitable for educational and assistive purposes.

## Key Features

*   **Real-Time Recognition**: Utilizes webcam feed to recognize ASL signs instantly.
*   **MediaPipe Integration**: Extracts 21 3D hand landmarks, significantly reducing computational complexity compared to image-based methods.
*   **MLP Classifier**: A Multi-Layer Perceptron model is trained on the extracted landmarks for robust classification.
*   **Interactive Learning Interface**: Features a Tkinter GUI for live training feedback, displaying target letters, predictions, and scores.
*   **Efficient Performance**: Achieves high accuracy (best validation accuracy: 95.47%) with fast inference, making it practical for live applications.

## How it Works

1.  **Hand Landmark Extraction**: MediaPipe Hands processes live video input to detect and extract 21 3D landmarks from the user's hand.
2.  **Feature Input**: These 63 landmark coordinates (x, y, z for 21 points) are flattened and fed as input to the MLP model.
3.  **MLP Classification**: The trained MLP model, consisting of an input layer, two hidden layers with LeakyReLU activation and Dropout, and a Softmax output layer, predicts one of the 26 ASL alphabet classes.
4.  **Real-Time Feedback**: The system displays the predicted letter, provides feedback, and tracks the user's score in real-time.

## Dataset & Training

*   **Dataset**: A cleaned CSV file containing hand landmark coordinates for ASL signs (A-Z) was used.
*   **Preprocessing**: Only uppercase English alphabet labels were retained. Features were landmark coordinates, and labels were one-hot encoded.
*   **Split**: Data was split into 80% training and 20% testing sets.
*   **Model**: A TensorFlow/Keras MLP model was built and trained.
*   **Optimizer & Loss**: Adam optimizer and `sparse_categorical_crossentropy` loss function were used.
*   **Custom Callback**: A custom callback saved the model only when both validation accuracy increased and validation loss decreased, ensuring optimal model saving.

## Setup and Usage

To run this project, you will need Python, OpenCV, MediaPipe, and TensorFlow. The `learn_asl_live.py` script handles the real-time recognition and interactive GUI. The `train_hand_mlp_cleaned.ipynb` notebook contains the training process for the MLP model.



