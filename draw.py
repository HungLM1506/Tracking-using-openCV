from opt import *

img_path = r'back_ground.jpg'

img = cv2.imread(img_path)

roi = cv2.selectROI('image', img, showCrosshair=False, fromCenter=False)

cv2.imshow('image', img)
cv2.waitKey()

f = open('Spot_warning.txt', 'w+')
x, y, w, h = roi
f.write(f'{x},{y},{w},{h}')
f.close()
