import tkinter as tk
import time
import cv2
import dlib
import os
import numpy as np
import subprocess
import threading




def update_timer():
    root.after(0, update_interface)
    global seconds, minutes, hours
    seconds += 1
    if seconds >= 60:
        seconds = 0
        minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1

    timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    if timer_running:
        timer_canvas.after(1000, update_timer)
        timer_canvas.itemconfig(timer_text, text=timer_var.get())
    

def start_timer():
    global timer_running
    if not timer_running:
        timer_running = True
        update_timer()

def stop_time():
    global timer_running
    timer_running = False
    

def reset_timer():
    global seconds, minutes, hours
    seconds, minutes, hours = 0, 0, 0
    timer_var.set("00:00:00")
    timer_canvas.itemconfig(timer_text, text=timer_var.get())




recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")



def calculate_distance(descriptor1, descriptor2):
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))



def stop_timer():
    try:
        subprocess.run(["osascript", "-e", 'tell application "System Events" to sleep'])
        print("Display has been put to sleep.")
    except Exception as e:
        print("Error putting display to sleep:", e)




def capture_and_store_face_data():
    # Initialize the camera
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




def check_face_recognition():
    camera = cv2.VideoCapture(0)
    user_detected = False

    while True:
        time.sleep(1)
        ret, frame = camera.read()
        if not ret:
            print("Error capturing an image.")
            stop_time()
            continue

        detector = dlib.get_frontal_face_detector()
        faces = detector(frame)

        if not faces:
            if user_detected:
                print("User not in front of the camera. Putting the display to sleep.")
                global seconds
                if timer_running == True: seconds-=1
                stop_time()
                user_detected = False
            else:
                print("No face detected. Trying again...")
                stop_time()
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
                start_timer()
                user_detected = True
            else:
                if user_detected:
                    print("User not recognized.")
                    if timer_running == True: seconds-=1
                    stop_time()
                    user_detected = False
                else:
                    print("User not recognized. Continuing face recognition...")
                continue  # Continue recognizing faces

        else:
            print("No user data found. Please run the program to capture your face data.")
            continue  # Continue recognizing faces
        


def update_interface():
    # Update the tkinter interface here
    root.update() 




window = tk.Tk()
window.title("Timer")
window.geometry("700x586")  # Adjust the window size to accommodate the buttons
window.config(bg="black")
seconds, minutes, hours = 0, 0, 0
timer_running = False

timer_var = tk.StringVar()
timer_var.set("00:00:00")

timer_canvas = tk.Canvas(window, height=486, width=700, highlightthickness=0)
timer_img = tk.PhotoImage(file="timer.png")
timer_canvas.create_image(350, 243, image=timer_img)
timer_text = timer_canvas.create_text(350, 243, text=timer_var.get(), fill="white", font=("Courier", 130, "bold"))
timer_canvas.grid(row=0, column=0, columnspan=3)

start_button = tk.Button(window, text="Start",highlightthickness=0, highlightbackground="black", command=start_timer)
stop_button = tk.Button(window, text="Store Data",highlightthickness=0, highlightbackground="black", command=capture_and_store_face_data)
reset_button = tk.Button(window, text="Reset", highlightthickness=0, highlightbackground="black", command=reset_timer)

start_button.grid(row=1, column=0,pady=30, sticky='n')
stop_button.grid(row=1, column=1)
reset_button.grid(row=1, column=2)

root = window  # Store the root window for the update

# Create a separate thread for the infinite loop function
infinite_loop_thread = threading.Thread(target=check_face_recognition)
infinite_loop_thread.daemon = True  # This will allow the program to exit when the GUI window is closed
infinite_loop_thread.start()

window.mainloop()
