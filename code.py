import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\Khouloud\Desktop\demo_Model\best.pt')
model.eval()
model.conf = 0.4  # set confidence threshold

#cap = cv2.VideoCapture('C:/Users/Khouloud/Desktop/demo_Model/fight_students_school.mp4')
#cap = cv2.VideoCapture('C:/Users/Khouloud/Desktop/demo_Model/school_fight_student.mp4') 
cap = cv2.VideoCapture('C:/Users/Khouloud/Desktop/demo_Model/fighting_no_training_in.mp4')
while True:
    ret, frame = cap.read()  # Read frame from video stream
    if not ret:
        break
    results = model(frame)
    print(results.render()[0])  # create a frame in the object I want to detect
    conf = results.pandas().xyxy[0]['confidence'].tolist()
    name = results.pandas().xyxy[0]['name'].tolist()
    print(conf)
    print(name)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
