import cv2
import numpy as np
from skimage import io


image = cv2.imdecode(np.fromfile('./wallpaper knife/xray/01133.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
cv2.imshow('ori', image)


#-------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#-------------
kernel = np.ones((9, 9), np.uint8)
dilation = cv2.dilate(binary, kernel)
contours, hierarchy = cv2.findContours(dilation,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)


#-------------
x, y, w, h = cv2.boundingRect(contours[0])
hull = cv2.convexHull(contours[0])

brcnt = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
cv2.drawContours(image, [brcnt], -1, (0, 0, 0), 2)
cv2.polylines(image, [hull], True, (0, 255, 0), 2)
print(hull)

#-------------
cv2.imshow("result", image)
cv2.waitKey()
cv2.destroyAllWindow()
