from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import cv2

with open('pose_model.pkl','rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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
    return pose

actions = np.array(['idle','holding_controller', 'texting', 'stretch', 'left_raise_full', 'left_raise_half', 'right_raise_full', 'right_raise_half', 'left_hand_on_desk', 'right_hand_on desk', 'talking_left', 'talking_right'])

cap = cv2.VideoCapture(0)
sequence_length = 300
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)     
        
        # read keypoints
        try:
            pose_row = list(extract_keypoints(results))
            X = pd.DataFrame([pose_row])
            pose_class = model.predict(X)[0]
            pose_prob = model.predict_proba(X)[0]
            print(pose_class, pose_prob)
            coords = (15,30)
            cv2.rectangle(image,
                          (coords[0], coords[1]+10),
                          (coords[0]+len(pose_class)*20, coords[1]-30),
                          (245,117,16),-1)
            cv2.putText(image, 
                        pose_class, 
                        (coords[0]+5,coords[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,255,255),2,cv2.LINE_AA) 
        except:
            print('excepting')
            pass

        cv2.imshow('Pose Detection', image)
        
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
                    
    cap.release()
    cv2.destroyAllWindows()


        

        



