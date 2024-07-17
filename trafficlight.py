import cv2
import numpy as np

# 색상 범위 설정 (HSV)
# 빨간색 범위 (두 범위로 나누어 설정)
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

# 노란색 범위
yellow_lower = np.array([20, 120, 70])
yellow_upper = np.array([30, 255, 255])

# 녹색 범위
green_lower = np.array([40, 70, 70])
green_upper = np.array([80, 255, 255])

def detect_traffic_light_color(image, roi):
    # 관심 영역 설정
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]
    
    # 이미지를 HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    
    # 빨간색 마스크 생성 (두 범위 합침)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 노란색 마스크 생성
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # 녹색 마스크 생성
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 각각의 마스크에서 색상 픽셀 수 계산
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    # 가장 많은 픽셀이 검출된 색상을 반환
    if red_count > yellow_count and red_count > green_count:
        return 'Red'
    elif yellow_count > red_count and yellow_count > green_count:
        return 'Yellow'
    elif green_count > red_count and green_count > yellow_count:
        return 'Green'
    else:
        return 'Unknown'

# 테스트 이미지 로드
image = cv2.imread('green.jpg')

# 관심 영역 설정 (x, y, width, height)
roi = (75, 125, 225, 100)  # 예: 이미지의 중간 부분

# 관심 영역을 사각형으로 표시
cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

# 신호등 색상 인식
color = detect_traffic_light_color(image, roi)
print(f'Detected traffic light color: {color}')

# 결과 이미지 출력
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
