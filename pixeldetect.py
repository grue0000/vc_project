import cv2
import matplotlib.pyplot as plt
import numpy as np

for i in range (30):
    image = cv2.imread('Images/' + str(i + 1) + '.bmp', cv2.IMREAD_GRAYSCALE)
    sze = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    dst = cv2.GaussianBlur(sze, (5, 5), sigmaX=0)
    dst = cv2.equalizeHist(dst)
    edge1 = cv2.Canny(dst, 90, 100)
    print(edge1.shape)
    roi = edge1
    stepSize = 5
    (w_width, w_height) = (87, 72)
    pixsum = 255 * w_width * w_height
    for x in range(0, roi.shape[1] - w_width, stepSize):
        for y in range(0, roi.shape[0] - w_height, stepSize):
            window = roi[x:x + w_width, y:y + w_height]
            if pixsum > window.sum() > 0 :  # pixsum에 가장 초기의 값
                pixsum = window.sum()
                minxy = [x, y]  # 여기 부분에서 문제가 생겼다고 생각 x좌표 고정됨 다른 사진의 경우 고정되지 않음
            print(minxy)
    print(pixsum)
    print(minxy)
    [x, y] = minxy
    cv2.rectangle(roi, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
    # plt.imshow(np.array(tmp).astype('uint8'))
    # plt.show()
    cv2.imshow('IMG', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()