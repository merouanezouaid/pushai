# import cv2
# import mediapipe as mp
# import numpy as np

# data = np.load("dataset/labels/correct.npy")

# #print(data)

# cap = cv2.VideoCapture("dataset/Correct sequence/Copy of push up 1.mp4")

# mp_drawings = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# pose_landmarks_list = []


# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  
#     while cap.isOpened():
#         ret, image = cap.read()
#         if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)

#         if results.pose_landmarks:
#             pose_landmarks_list.append(results.pose_landmarks)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
#                                 mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#                                 mp_drawings.DrawingSpec(color=(245,0,230), thickness=4, circle_radius=2)
#         )

#         cv2.imshow('Raw Webcam Feed', image)

#         if(cv2.waitKey(10) & 0xFF == ord('q')):
#             break

# cap.release()
# cv2.destroyAllWindows()

# pose_landmarks_list = np.array(pose_landmarks_list)
# print(pose_landmarks_list)
# np.save("dataset/labels/positions.npy", pose_landmarks_list)



import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load drawing utils and pose solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load your trained LSTM model
model = load_model("model.h5")

def preprocess_frame(frame):
    # Convert BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe pose object
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Process frame with MediaPipe pose
        results = pose.process(rgb_frame)

        # Extract keypoints if a pose is detected
        if results.pose_landmarks:
            # Select desired keypoints 
            keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten()

            # Assuming you want 100 frames (timesteps)
            timesteps = 100  # Adjust based on your needs

            # Check if you have enough frames
            if keypoints.shape[0] < timesteps:
                # Pad with zeros if there are less than desired frames
                keypoints = np.pad(keypoints, ((0, timesteps - len(keypoints))), mode='constant')
            else:
                # Select a window of the desired number of frames (optional)
                keypoints = keypoints[:timesteps]  # Select the first 100 frames

            # Reshape to expected format (assuming 3 features per keypoint)
            preprocessed_frame = keypoints.reshape((1, timesteps, -1))  # (samples, timesteps, features)


            # print("Shape before padding:", preprocessed_frame.shape)
            # Pad with zeros to match the desired shape (if necessary)
            preprocessed_frame = np.pad(preprocessed_frame, ((0, 0), (0, 0), (0, 99 - preprocessed_frame.shape[2])), mode='constant')

            # print("Shape after padding:", preprocessed_frame.shape)

        else:
            # Handle case where no pose is detected 
            preprocessed_frame = np.empty((1, timesteps, 99))  # Placeholder for empty frame with 3 features

    return preprocessed_frame


# Open video capture
# cap = cv2.VideoCapture("dataset/Correct sequence/Copy of push up 187.mp4")
cap = cv2.VideoCapture("dataset/Wrong sequence/9.mp4")

# Get video frame dimensions 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: Video writer for output (replace codec and dimensions if needed)
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

while True:
  # Read frame
    ret, frame = cap.read()
    if not ret:
        break


    # Preprocess frame 
    preprocessed_frame = preprocess_frame(frame.copy())

    # Predict push-up classification
    prediction = model.predict(preprocessed_frame)
    print(prediction)
    classification = "Correct" if prediction[0][0] > 0.5 else "Incorrect"

    # Display prediction on frame
    cv2.putText(frame, classification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Optional: Draw pose landmarks on frame (if desired)
    # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display frame
    cv2.imshow("Push-up Classification", frame)

    # Optional: Write frame to output video
    #writer.write(frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
# if writer is not None:
#     writer.release()
cv2.destroyAllWindows()
