import cv2
import dlib
import os
import numpy as np
import subprocess

"""
As seen in this activity diagram of our project we have shown the basic implementation of the two 
function with the name capture_and_store_face_data() and other with the name check face recognition. 
using these functions we are performing the operation of capturing and recognition of the image

At first in the function capture and store face data we are using the open cv module by using import cv2
.We are then using this to capture the current image in front of the camera.
After then we are using the data to read the image. 
after reading the image we are using the dlib to ge the frontal face. 
dlib has a function known as frontal face detector. 
In this function we are passing the frames which we recorded with the help of the open cv2.
After that we are using the shape_predictor_68_face_landmarks.
dat to determine whether the image detected has a face or not.
If there is no face detected in the image then the program returns message that their 
is no face in front of the camera and if it detects that there is a face in front of the 
camera then it stores the data of that face in a text file.
Then in the function check face recognition we are using the camera to capture the image using opencv 
after that we are running an infinite loop to examine the face data. If the examination gives the result
that the image in front of camera has the user with the same face data stored using the above function 
then the loop continuous. If it is determined that the user in front of the camera is another person then
the program uses the subprocess module to give the operating system command to go to sleep. 
And if there is no user determined in front of the camera then also the program commanda the operating 
system to fo to sleep.This is done with the help of initialization a recognizer object which uses a 
black box model dlib_face_recognition_resnet_model_v1.dat to determine the image. And we also calculate 
the distance for it so that we can say that the user is in the appropriate distance to be recognised by 
our program in our case we have taken it to be 0.6.
"""


# Path to the trained face recognition model
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to calculate the Euclidean distance between two face descriptors
def calculate_distance(descriptor1, descriptor2):
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))

# Function to put the display to sleep
def put_display_to_sleep():
    try:
        subprocess.run(["osascript", "-e", 'tell application "System Events" to sleep'])
        print("Display has been put to sleep.")
    except Exception as e:
        print("Error putting display to sleep:", e)

# Function to capture and store user's face data
def capture_and_store_face_data():
    camera = cv2.VideoCapture(0)

    # Try for a number of frames before concluding that no face is detected
    max_attempts = 20  # Adjust as needed
    attempts = 0

    while attempts < max_attempts:
        # Capture a single frame
        ret, frame = camera.read()

        if not ret:
            print("Error capturing an image.")
            continue

        # Detect faces in the captured frame
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame)

        if not faces:
            attempts += 1
            continue

        # Get facial landmarks
        landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_descriptor = recognizer.compute_face_descriptor(frame, landmarks(frame, faces[0]))

        # Save the user's face data to a file
        with open("user_face_data.txt", "w") as file:
            for value in face_descriptor:
                file.write(str(value) + " ")

        print("Face data stored successfully.")
        break

    if attempts == max_attempts:
        print("No face detected in multiple attempts. Exiting.")
        put_display_to_sleep()

# Function to check if the user's face matches the stored data
def check_face_recognition():
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    user_detected = False  # Track whether the user was last detected

    while True:
        # Capture a single frame
        ret, frame = camera.read()

        if not ret:
            print("Error capturing an image.")
            continue

        # Detect faces in the captured frame
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame)

        if not faces:
            if user_detected:
                print("User not in front of the camera. Putting the display to sleep.")
                put_display_to_sleep()
                user_detected = False
            else:
                print("No face detected. Trying again...")
            continue

        # Get facial landmarks
        landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_descriptor = recognizer.compute_face_descriptor(frame, landmarks(frame, faces[0]))

        # Load the stored user's face data
        if os.path.exists("user_face_data.txt"):
            with open("user_face_data.txt", "r") as file:
                stored_face_data = [float(value) for value in file.read().split()]

            # Compare the stored data with the current data
            distance = calculate_distance(stored_face_data, face_descriptor)
            
            # Adjust the threshold as needed (lower values indicate a better match)
            threshold = 0.6

            if distance < threshold:
                print("Welcome back, user.")
                user_detected = True
            else:
                if user_detected:
                    print("User not recognized. Putting the display to sleep.")
                    put_display_to_sleep()
                    user_detected = False
                else:
                    print("User not recognized. Continuing face recognition...")
                continue  # Continue recognizing faces

        else:
            print("No user data found. Please run the program to capture your face data.")
            continue  # Continue recognizing faces

# Main program
if __name__ == "__main__":
    while True:
        choice = input("Enter 'c' to capture face data, 'r' to recognize, or 'q' to quit: ")
        if choice == 'c':
            capture_and_store_face_data()
        elif choice == 'r':
            check_face_recognition()
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please enter 'c', 'r', or 'q'.")
