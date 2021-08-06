import numpy as np
import cv2
import glob

def drawchecker(pic,corners):

    for i in range(7):
        pic = cv2.line(pic, tuple(corners[i][0]), tuple(corners[i+42][0]), (255,0,0),2)
        pic = cv2.line(pic, tuple(corners[i*7][0]), tuple(corners[i*7+6][0]), (255,0,0),2)
    return pic

# 퍼터 찾기
def findputter(img):
        #roi = img[90: 380, 300: 550]
        # dst = cv2.GaussianBlur(roi, (5, 5), sigmaX=0)
        try:
            circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=400, param2=11, minRadius=5, maxRadius=30)

            for k in circles[0]:
                cv2.circle(roi, (int(k[0]), int(k[1])), int(k[2]), (0, 0, 0), 5)  # 중심점,  반지름
            # print(circles[0])
            # print(circles[0][0][0])
            # print(circles[0][0][1])
            # print(circles[0][1][0])
            # print(circles[0][1][1])
                x1 = circles[0][0][0]
                y1 = circles[0][0][1]
                x2 = circles[0][1][0]
                y2 = circles[0][1][1]
        # 속도, 궤적 구하는 부분
            # cenX = int((x1 + x2) / 2)
            # cenY = int((y1 + y2) / 2)
            # pointarr.append([cenX, cenY])
            # pts = np.array(pointarr, np.int32)
            # print(pts)
            # roi = cv2.polylines(roi, [pts], False, (0, 255, 0), 2)
            return roi
        except:
            return img

def ROI(img, verticles):
    # sze = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # dst = cv2.GaussianBlur(image, (5, 5), sigmaX=0)
    # dst = cv2.equalizeHist(dst)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [verticles], (255, 255, 255))
    roi_image = cv2.bitwise_and(img, mask)
    return roi_image

# 웹캠 연결
cap = cv2.VideoCapture(1)

# 템플릿 이미지
template = cv2.imread("tableball.jpg",cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, None, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
#template1 = cv2.imread("whitebox.jpg")
# termination criteria
garo = 9
sero = 6
flag = 1
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((garo * sero,3), np.float32)
objp[:,:2] = np.mgrid[0:sero,0:garo].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

lx = 0
ly = 0

# 왜곡계수, 카메라 매트릭스
mtx = [[6.18775659e+03, 0.00000000e+00, 2.44691923e+02], [0.00000000e+00, 9.32463679e+03, 2.30735248e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist = [[7.43442239e+00, 4.51858581e+03, -1.75821013e-01, -1.24429521e-01, 2.59065570e+00]]
mtx = np.array(mtx)
dist = np.array(dist)

verticles = np.array([[0,0],[0,0],[0,0],[0,0]], np.int32)
cmperpix = 0

pointarr = []

while(True):
    ret, img = cap.read()    # Read 결과와 frame

    if(ret) :

      # 왜곡보정
        h, w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        img1 = dst[y:y+h, x:x+w]

      # grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 매트 인식부
        if flag:
          # 흰색 마커 찾기 (canny edge detection)
            edge1 = cv2.Canny(img1, 290, 300)
            contours, _ = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            buffer = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                if 15 < w < 40 and 15 < h < 45:
                    if gray1[x:x + w, y:y + h].mean() < 250:
                        buffer.append([abs(y - 320), x, y, w, h])  # 사진의 중앙에서 가장 가까운 흰색 박스 6개 찾음
            buffer = sorted(buffer)[:6]
            for lst in buffer:
                _, x, y, w, h = lst
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            buffer = sorted(buffer, key=lambda buffer: buffer[1])
            leftdown = [buffer[0][1]+buffer[0][3], buffer[0][2]]
            leftup = [buffer[2][1]+buffer[2][3], buffer[2][2]]
            rightdown = [buffer[5][1], buffer[5][2]]
            rightup = [buffer[3][1], buffer[3][2]]

            verticles = np.array([leftdown, leftup, rightup, rightdown], np.int32) # roi 좌표
            cmperpix = 40.1/(rightup[1]-rightdown[0]) # 1 픽셀당 실제거리(cm)
            if cv2.waitKey(1) == ord('p'):
                flag = 0

        else:
            # roi 지정
            roi_image = ROI(img1, verticles)
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

            # 골프공 템플릿 매칭
            # result = cv2.matchTemplate(roi_gray, template, cv2.TM_SQDIFF_NORMED)
            # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            # x, y = minLoc
            # h, w = template.shape
            # if result.mean() < 0.985:
            #     cenX = x+h/2
            #     cenY = y+w/2
            #     pointarr.append([cenX, cenY])
            #     dst = cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # pts = np.array(pointarr, np.int32)
            # dst = cv2.polylines(dst, [pts], False, (0, 255, 0), 2)

            # 골프공 canny edge detection
            edge1 = cv2.Canny(roi_image, 250, 300)
            contours, _ = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            buffer = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 43 < w < 70 and 43 < h < 70:
                    print(w,h)
                    # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cenX = x + h / 2
                    cenY = y+w/2
                    pointarr.append([cenX, cenY])
            pts = np.array(pointarr, np.int32)
            #img1 = cv2.polylines(img1, [pts], False, (0, 255, 0), 2)

            # 퍼터 찾기
            img1 = findputter(img1)




        #cv2.imshow('frame', gray)
            cv2.imshow('edge1', edge1)
        #cv2.imshow('before', img)    # 보정 전 화면 출력
        cv2.imshow('after', img1)   # 보정 후 화면 출력
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

