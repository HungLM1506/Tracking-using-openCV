from email import iterators
from unittest import result
import cv2
import numpy as np
import math
from collections import defaultdict
center_points = {}
id_count = 0
def update(object_rect):
    boxs_id = []
    for rect in object_rect:
        x, y, w, h = rect
        center_x = x + w//2
        center_y = y + h//2
        detected = False
        global center_points, id_count
        for (id, point) in center_points.items():
            distance = math.hypot(center_x - point[0], center_y - point[1])
            if distance < 25:
                center_points[id] = (center_x, center_y)
                boxs_id.append([x,y,w,h,id])
                detected = True
                break
        if detected is False:
            center_points[id_count] = (center_x, center_y)
            boxs_id.append([x, y, w, h, id_count])
            id_count += 1
    new_center_points = {}
    for box in boxs_id:
        _, _, _, _, object_id = box
        center = center_points[object_id]
        new_center_points[object_id] = center
    center_points = new_center_points.copy()
    return boxs_id        

points = []
        
def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

       
def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,255,0), -1)
    frame = cv2.polylines(frame,[np.int32(points)], False, (255,0, 0), thickness = 2)
   
    return frame
def check ( points, centroid):
    points = np.array(points,dtype=np.int32 )
    points = points.reshape((-1, 1, 2))
    # print(points)
    x = cv2.pointPolygonTest(points,centroid,False )
    return x
list_centroid = defaultdict(list)
count = 0
data = []
centroid_list = []
list_id = []
def draw(frame, x, y, w, h, id, points):
    global count, data, centroid_list 
    cv2.putText(frame, str(id),(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
    #cv2.circle(frame, ((x+w//2),(y + h//2)), 3, (255,0,255), 3)
    centroid = ((x + w//2),(y+h//2)) 
    points = np.array(points,dtype=np.int32 )
    points = points.reshape((-1, 1, 2)) 
    centroid_list.append(centroid)
    list_id.append(id)
    #dem nguoi trong vung nguy hiem    
    if check(points, centroid) == 1:
       # print(count)
        cv2.fillPoly(frame, [points], (0, 0, 255))
        
        if id not in data :
            count += 1
            data.append(id)
        
        if id in data and check(points, centroid_list[id]) == -1:
                data.remove(id)  
                count -= 1
                print(data)
        
           # print(data)
        #cv2.imshow('opacity', frame)
    cv2.putText(frame, str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 3)
    #print(points
    
    
COLORS = np.random.randint(0, 255, (200, 3))
cap = cv2.VideoCapture('299988101_5785136164852869_7477433635271370041_n.mp4')
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (1280,720))
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray1, (5,5), 0)
detect = False
centroid_dict = defaultdict(list)
object_id_list = []
while True:
    ret, frame = cap.read()
    img = frame.copy()
    frame = cv2.resize(frame, (1280,720))
    img = frame.copy()
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray2, (5,5), 0)
    diff = cv2.absdiff(blur, blur1)
    cv2.imshow('diff', diff)
    _, thresh = cv2.threshold(diff, 23,255,cv2.THRESH_BINARY)
    contours, _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    detections = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) <300:
            continue
        detections.append([x,y,w,h])
    boxes_ids = update(detections)
    x = draw_polygon(frame, points)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        centroid = ((x + w//2),(y+h//2)) 
        draw(frame, x, y, w, h, id , points)
        color = [int(c) for c in COLORS[ id % len(COLORS)]]
        cX = x + w//2
        cY = y + h//2
        centroid_dict[id].append((cX, cY))
        if id not in object_id_list:
            object_id_list.append(id)
            start_pt = (cX, cY)
            end_pt = (cX, cY)
            cv2.line(frame, start_pt, end_pt, (color), 3)
        else:
            l = len(centroid_dict[id])
            for pt in range(len(centroid_dict[id])):
                if not pt + 1 == l:
                    start_pt = (centroid_dict[id][pt][0], centroid_dict[id][pt][1])
                    end_pt = (centroid_dict[id][pt + 1][0], centroid_dict[id][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, (color), 3)     
    frame= cv2.addWeighted(img,0.35, frame, 0.65, 0)   
    cv2.imshow('frame',frame)   
    cv2.setMouseCallback('frame',handle_left_click, points)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    if key == ord('d'):
        points.append(points[0])
        print(points)
        detect = True  
cap.release()
cv2.destroyAllWindows()

    
    