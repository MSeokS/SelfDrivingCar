import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """허프 변환을 사용하여 직선 검출"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = average_slope_intercept(img, lines)
    print(lines)
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """검출된 직선을 이미지에 그림"""
    if lines is None:
        return
    for line in lines:
        [x1, y1, x2, y2] = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line_parameters):
    height, width = frame.shape
    
    slope, intercept = line_parameters
    
    y1 = height
    y2 = int(height * 0.6)
    
    if slope != 0:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        x1 = x2 = 0
    
    points = [x1, y1, x2, y2]
    
    return points

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


img = cv2.imread("test_image.jpg")

# 그레이스케일 변환
gray = grayscale(img)

# 가우시안 블러 적용
blur_gray = gaussian_blur(gray, 5)

# Canny 엣지 검출 적용
edges = canny(blur_gray, 50, 150)

# 관심 영역 설정
imshape = img.shape
print(imshape)
vertices = np.array([[
    (imshape[1] * 0, imshape[0] * 0.95),          # 아래 왼쪽
    (imshape[1] * 0.4, imshape[0] * 0.6),   # 위쪽 왼쪽
    (imshape[1] * 0.6, imshape[0] * 0.6),   # 위쪽 오른쪽
    (imshape[1] * 1.0, imshape[0] * 0.95)           # 아래 오른쪽
]], dtype=np.int32)

mask = np.zeros_like(img)
ignore_mask_color = 255

cv2.fillPoly(mask, vertices, ignore_mask_color)

roi_image = weighted_img(img, mask)
cv2.imshow('Lane Detection', roi_image)
cv2.waitKey(0)

masked_edges = region_of_interest(edges, vertices)

# 허프 변환을 사용하여 직선 검출
line_image = hough_lines(masked_edges, 1, np.pi/180, 15, 40, 20)

# 원본 이미지에 직선 이미지를 합침
lines_edges = weighted_img(line_image, np.zeros(imshape, dtype=np.uint8))

# 결과 출력
cv2.imshow('Lane Detection', lines_edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
