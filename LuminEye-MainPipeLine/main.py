import cv2
import dlib
from imutils import face_utils
import hpe
import numpy as np
import mediapipe


# DLib Prediction
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Set True to Run the Pipeline from Dlib
dlib = False


# MediaPipe Prediction
mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)



# Read video feed from webcam
cv2.namedWindow('Webcam')
vid = cv2.VideoCapture(0)

if vid.isOpened():
    
    val, frame = vid.read() 
else:
    val = False

frameCounter = 0

while val:

    # Read BGR image frame
    val, frame = vid.read()

    # Keep a count of the frames
    frameCounter += 1
    
    
    if dlib:

        # Convert image frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in grayscale image frame
        rects = detector(gray, 1)

        # Loop over the face detections
        for (i, rect) in enumerate(rects):

            # Display a bounding box for each face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if w * h > 2000:

                # Detect the face landmarks
                shape = predictor(gray, rect)
                shape_arr = face_utils.shape_to_np(shape)

                # Display the face landmarks
                for (m, n) in shape_arr:
                    cv2.circle(frame, (m, n), 1, (0, 0, 255), -1)

                # cv2.circle(frame, (shape_arr[33, :]), 1, (0, 255, 255), -1)
                
    else:
        results = face_mesh.process(frame)
        landmarks = results.multi_face_landmarks[0]
        
        
        shape_arr = []

        for landmark in landmarks.landmark:
            x = landmark.x
            y = landmark.y
            
            relative_x = int(x * frame.shape[1])
            relative_y = int(y * frame.shape[0])
            
            
            cv2.circle(frame,(relative_x,relative_y),radius = 2,color=(0,255,0),thickness=-1)
            
            shape_arr.append([relative_x, relative_y])


        shape_arr = np.array(shape_arr)

    # Compute the rotation matrix from the measurement matrix
    r = hpe.compute_rotation(shape_arr)
    
    
    
    print(f"Rotation Matrix: {r}")

    # Compute the head rotation angles from the rotation matrix
    yaw, pitch, roll, yaw_deg, pitch_deg, roll_deg = hpe.compute_angles(r, frameCounter)
    
    print(f"Frame Counter: {frameCounter}")
    print(f"Yaw: {yaw} pitch:{pitch} roll:{roll}")

    # Write the yaw, pitch and yaw angles on the image
    cv2.putText(frame, str([np.round(yaw_deg), np.round(pitch_deg), np.round(roll_deg)]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw direction vector
    start_pt = shape_arr[14, :]
    end_pt = np.array([-(5 * np.int0(np.sin(yaw) * (180 / np.pi))) + start_pt[0],
                        (5 * np.int0(np.sin(pitch) * (180 / np.pi))) + start_pt[1]])
    cv2.arrowedLine(frame, start_pt, end_pt, color=(255, 0, 0), thickness=2)

    # Display image frame
    cv2.imshow('Webcam', frame)

    # Wait for key press
    key = cv2.waitKey(10)

    if key == 27:
        break

vid.release()
cv2.destroyWindow('Webcam')
