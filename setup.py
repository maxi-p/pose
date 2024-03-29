import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Path for exported data
DATA_PATH = os.path.join("MP_Data")
actions = np.array(['idle','holding_controller', 'texting', 'stretch', 'left_raise_full', 'left_raise_half', 'right_raise_full', 'right_raise_half', 'left_hand_on_desk', 'right_hand_on desk', 'talking_left', 'talking_right'])
no_sequences = 30
sequence_length = 30
start_folder = 0

# Setting up folders
for action in actions:
	for sequence in range(no_sequences):
		try:
			os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
		except:
			pass



def draw_landmarks(image, results):
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def extract_keypoints(results):
    i = 0
    arr = []
    if(results.pose_landmarks):
        for res in results.pose_landmarks.landmark:
            if i>10 and i<25:
                arr.append([res.x,res.y,res.z,res.visibility])
            i = i+1
        pose = np.array(arr).flatten()
    else:
        pose = np.zeros(14*4)   
    #pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(14*4)
    return np.concatenate([pose])
    # return np.concatenate([pose,lh,rh])
    
def mediapipe_detection(image,model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
	return image, results
    
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()