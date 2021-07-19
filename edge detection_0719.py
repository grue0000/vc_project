import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

img = cv2.imread('Images/charge.jpg', cv2.IMREAD_GRAYSCALE)

edge1 = cv2.Canny(img, 100, 200)
edge2 = cv2.Canny(img, 140, 200)
edge3 = cv2.Canny(img, 170, 200)

_, contours, hierachy= cv2.findContours(edge3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w>30 and h>30:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(img)
cv2.imshow('original', img)
cv2.imshow('Canny Edge1', edge1)
cv2.imshow('Canny Edge2', edge2)
cv2.imshow('Canny Edge3', edge3)

cv2.waitKey()
cv2.destroyAllWindows()

