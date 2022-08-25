import cv2
from tracker import *
from collections import defaultdict
import numpy as np

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("test.avi")
centroid_dict = defaultdict(list)
object_id_list = []
counter = 0

c = []
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
# using mog2
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    #total_frames = total_frames +1
    height, width, _ = frame.shape
    frame = cv2.resize(frame,(1280, 720))
    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]
    #print('Image Height       : ',height)
    #print('Image Width        : ',width)
    #print('Number of Channels : ',channels)

    # Extract ROI
    top_left, bottom_right = (400, 200), (1000, 700)
    roi = frame[200: 600,400: 1000]
    # 1. Object Detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.blur(gray_roi, (5, 5), 0)
    mask = object_detector.apply(gray_roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            print(len(cnt))
        


            detections.append([x, y, w, h])
            #print(detections)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    #print(boxes_ids)
    for box_id in boxes_ids:

        x1, y1, x2, y2 = box_id[0],box_id[1],box_id[2],box_id[3]
        #print(x1)
        #print(y1)
        objectID = box_id[4]
        color = [int(c) for c in COLORS[ objectID % len(COLORS)]]

        cX = int((x1+x1 + x2) / 2.0)
        cY = int((y1 +y1+ y2) / 2.0)
        cv2.circle(roi, (cX, cY), 4, (color), -1)
        if (cX>200 and cX<800):
            if (cY>400 and cY<1400):
             print("in ra")
             cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        centroid_dict[objectID].append((cX, cY))
        #print(objectID)
        object_id_list.append(objectID)
        #print(object_id_list)
        #print(len(centroid_dict[objectID]))
        counter = len(set(centroid_dict[objectID]))

        #print(object_id_list)
        if objectID not in object_id_list:
            object_id_list.append(objectID)
            #print(object_id_list)
            start_pt = (cX, cY)
            end_pt = (cX, cY)
            cv2.line(roi, start_pt, end_pt, (color), 2)
        else:
            l = len(centroid_dict[objectID])
            for pt in range(len(centroid_dict[objectID])):
                if not pt + 1 == l:
                    start_pt = (centroid_dict[objectID][pt][0], centroid_dict[objectID][pt][1])
                    end_pt = (centroid_dict[objectID][pt + 1][0], centroid_dict[objectID][pt + 1][1])
                    cv2.line(roi, start_pt, end_pt, (color), 2)

        
        cv2.putText(roi, str(objectID), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x1, y1), (x1 + x2, y1 + y2), (color), 3)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)
        #count = len(set(objectID))
        cv2.putText(frame, "Total Object Counter: "+str(counter),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        #print(len(set(object_id_list)))

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()