# 사진에서 골프공, 퍼터 인식 코드

import cv2
import numpy as np

fps = 899
t = 1 / fps
lx = 0
ly = 0

pointarr = []  # 골프공 중심 좌표 저장할 list

img = cv2.imread("D_899_1/temp-04302021112709-784.bmp", cv2.IMREAD_GRAYSCALE)  # 첫 번째 이미지(골프공 정지해있는 사진)
width = 0
edge1 = cv2.Canny(img, 100, 200)
contours, hierachy = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 30 and h > 30:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        width = w  # width가 골프공의 지름

for i in range(30):
    src = cv2.imread("D_899_1/temp-04302021112709-" + str(784 + i) + ".bmp", cv2.IMREAD_GRAYSCALE)
    templit = cv2.imread("ball.jpg", cv2.IMREAD_GRAYSCALE)
    templit1 = cv2.imread("putter1.jpg", cv2.IMREAD_GRAYSCALE)

    res = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)  # src-공 match template

    # 소스 이미지 이진화(사진 밝기에 따라 threshold값 바꿈)
    src1 = cv2.imread("D_899_1/temp-04302021112709-" + str(784 + i) + ".bmp")
    srcgray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    if src[239][719] < 100:  # 우측 하단 밝기값
        ret, srcdst = cv2.threshold(srcgray, 110, 255, cv2.THRESH_BINARY)
    else:
        ret, srcdst = cv2.threshold(srcgray, 85, 255, cv2.THRESH_BINARY)

    dst = cv2.imread("D_899_1/temp-04302021112709-" + str(784 + i) + ".bmp")

    res1 = cv2.matchTemplate(srcdst, templit1, cv2.TM_SQDIFF_NORMED)  # 이진화 된 src-드라이버 match template

    # 물체 위치 박싱
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    x, y = minLoc
    h, w = templit.shape

    cenX = int(x + w / 2)  # 골프공 중심
    cenY = int(y + h / 2)

    pointarr.append([cenX, cenY])
    pts = np.array(pointarr, np.int32)

    dst = cv2.polylines(dst, [pts], False, (0, 255, 0), 2)

    dst = cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)

    minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(res1)
    x1, y1 = minLoc1
    h1, w1 = templit1.shape
    dst = cv2.rectangle(dst, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 1)

    vx = int(((x - lx) / t) * (4.27 / width) * 3600 / 100000)
    vy = -int(((y - ly) / t) * (4.27 / width) * 3600 / 100000)

    print('x속도 : ', vx, 'km/h  y속도 : ', vy, 'km/h')

    lx, ly = x, y

    # cv2.imwrite(str(i)+'.jpg', dst)
    cv2.imshow("srcdst", srcdst)
    cv2.imshow("templit1", templit1)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()