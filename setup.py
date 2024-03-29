import mediapipe as mp
import cv2
import numpy as np

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

landmarks = ['class']
for val in range(33):
    if val > 10 and val < 25:
        landmarks += ['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]

actions = np.array(['idle','holding_controller', 'texting', 'stretch', 'left_raise_full', 'left_raise_half', 'right_raise_full', 'right_raise_half', 'left_hand_on_desk', 'right_hand_on desk', 'talking_left', 'talking_right'])

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

cap = cv2.VideoCapture(0)
sequence_length = 300
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for frame_num in range(sequence_length+1):
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {}'.format(action), (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)
                continue
            else: 
                cv2.putText(image, 'Collecting frames for {}'.format(action), (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
            
            
            
            # NEW Export keypoints
            try:
                pose_row = list(extract_keypoints(results))
                pose_row.insert(0,action)
                with open('coords.csv',mode='a', newline='') as f:
                    csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(pose_row)
            except:
                pass
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()


        

        



