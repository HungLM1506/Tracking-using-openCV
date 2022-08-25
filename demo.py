from opt import *
from tracker import *
from collections import defaultdict

# Khởi tạo 1 số các thông số
count = 0
tracker = EuclideanDistTracker()
cap = cv2.VideoCapture('video\demo.mp4')
kernel = np.ones((3, 3), np.uint8)
centroid_dict = defaultdict(list)
object_id_list = []

# object_detector = cv2.createBackgroundSubtractorMOG2(
#     history=100, varThreshold=40)
# Đọc vào backgroud
file_path = 'back_ground.jpg'
back_ground = cv2.imread(file_path)

# Đọc vào vi tri vùng cấm
f = open('Spot_warning.txt', 'r')
slot = f.readline().strip()
xx, yy, ww, hh = map(int, slot.split(','))

# Ghi video theo dang avi
# frame_width = int(cap.get(3))

# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('video/output.mp4', cv2.VideoWriter_fourcc('M',
#                       'J', 'P', 'G'), 10, (frame_width, frame_height))
# Đọc video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask = cv2.absdiff(frame, back_ground)
    # mask = object_detector.apply(frame)
    # _, thresh = cv2.threshold(mask, 10, 0, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.dilate(mask, kernel=kernel)
    mask = cv2.erode(mask, kernel=kernel)
    mask = cv2.threshold(mask, 50, 200, cv2.THRESH_BINARY)[1]
    mask = cv2.Canny(mask, 20, 150)
    roi = frame[:500, 600:1500]

    # 1.object detection
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    list_id_object = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 900:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            # cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(frame, (xx, yy), (xx+ww, yy+hh), (0, 255, 0), 2)
            # cv2.circle(frame, (int(x+w/2), int(y+h/2)), 2, (0, 0, 255), -1)
            detections.append([x, y, w, h])

    # 2. object tracking
    boxes_ids = tracker.update(detections)
    # print(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx, cy = int((2*x+w)/2), int((2*y+h)/2)
        object_id_list.append(id)
        centroid_dict[id].append((cx, cy))
        cv2.putText(frame, str(id), (x, y-15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        check_ID = False
        if xx < x < xx+ww and yy < y < yy+hh:
            cv2.rectangle(frame, (xx, yy), (xx+ww, yy+hh), (0, 0, 255), 2)
        if id not in object_id_list:
            object_id_list.append(id)
            start_point = (cx, cy)
            end_point = (cx, cy)
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        else:
            l = len(centroid_dict[id])
            for point in range(l):
                if not point+1 == l:
                    start_point = centroid_dict[id][point][0], centroid_dict[id][point][1]
                    end_point = centroid_dict[id][point +
                                                  1][0], centroid_dict[id][point+1][1]
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, 'Count: '+str(count), (15, 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    # out.write(frame)
    cv2.imshow('Roi', roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
