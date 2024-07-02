from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.filters import threshold_otsu

def process_and_convert_image(image_path):
    img = Image.open(image_path)
    writable_img = img.copy()
    img_data = writable_img.load()
    width, height = writable_img.size

    for x in range(width):
        for y in range(height):
            pixel = img_data[x, y]
            if pixel == 0:
                img_data[x, y] = 255  # Black to white
            elif pixel == 255:
                img_data[x, y] = 180  # White to gray

    # Convert to numpy array for further processing
    img_np = np.array(writable_img)
    return img_np

def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [img[x_1][y], img[x_1][y1], img[x][y1],  img[x1][y1],     # P2,P3,P4,P5
            img[x1][y],  img[x1][y_1], img[x][y_1], img[x_1][y_1]]   # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]                            # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

# From https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
def zhang_suen_thinning(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()    # deepcopy to protect the original image
    changing1 = changing2 = 1       #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape         # x for rows, y for columns
        for x in range(1, rows - 1):                # No. of  rows
            for y in range(1, columns - 1):         # No. of columns
                n = neighbours(x, y, Image_Thinned)
                P2, _, P4, _, P6, _, P8, _ = n
                if (Image_Thinned[x][y] == 1 and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6 and            # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and         # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and          # Condition 3   
                    P4 * P6 * P8 == 0):             # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0

        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                n = neighbours(x, y, Image_Thinned)
                P2, _, P4, _, P6, _, P8, _ = n
                if (Image_Thinned[x][y] == 1 and    # Condition 0
                    2 <= sum(n) <= 6 and            # Condition 1
                    transitions(n) == 1 and         # Condition 2
                    P2 * P4 * P8 == 0 and           # Condition 3
                    P2 * P6 * P8 == 0):             # Condition 4
                    changing2.append((x,y))
        for x, y in changing2: 
            Image_Thinned[x][y] = 0

    return Image_Thinned

# Optimized version from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
def zhang_suen_thinning_optimized(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()     # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:    #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        white_pixels_indices = np.nonzero(Image_Thinned == 1)

        for (x, y) in np.array(white_pixels_indices).T:
            n = neighbours(x, y, Image_Thinned)
            P2, _, P4, _, P6, _, P8, _ = n
            if (Image_Thinned[x][y] == 1 and    # Condition 0: Point P1 in the object regions 
                2 <= sum(n) <= 6   and          # Condition 1: 2<= N(P1) <= 6
                transitions(n) == 1 and         # Condition 2: S(P1)=1  
                P2 * P4 * P6 == 0  and          # Condition 3   
                P4 * P6 * P8 == 0):             # Condition 4
                changing1.append((x,y))

        if len(changing1) > 0:
            changing1_indecies = np.array(changing1)
            Image_Thinned[changing1_indecies[:,0], changing1_indecies[:,1]] = 0

        # Step 2
        changing2 = []
        white_pixels_indices = np.nonzero(Image_Thinned == 1) 

        for (x, y) in np.array(white_pixels_indices).T:
            n = neighbours(x, y, Image_Thinned)
            P2, _, P4, _, P6, _, P8, _ = n
            if (Image_Thinned[x][y] == 1 and    # Condition 0: Point P1 in the object regions 
                2 <= sum(n) <= 6   and          # Condition 1: 2<= N(P1) <= 6
                transitions(n) == 1 and         # Condition 2: S(P1)=1  
                P2 * P4 * P8 == 0  and          # Condition 3   
                P2 * P6 * P8 == 0):             # Condition 4
                changing2.append((x,y))  

        if len(changing2) > 0:
            changing2_indecies = np.array(changing2)
            Image_Thinned[changing2_indecies[:,0], changing2_indecies[:,1]] = 0

    return Image_Thinned


def find_intersections(skeleton_image):
    # Находим белые точки на утонченном изображении
    skeleton_image_white_pixels_indices = np.array(np.nonzero(skeleton_image == 1)).T

    # Маска для подсчета соседей по вертикали и горизонтали
    mask_hv = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]],
                      dtype=np.uint8)
    
    # Маска для подсчета соседей по диагонали
    mask_diag = np.array([[1, 0, 1],
                      [0, 0, 0],
                      [1, 0, 1]],
                      dtype=np.uint8)

    # Пустые массивы для количества соседей
    neighbors_hv = np.zeros(skeleton_image_white_pixels_indices.shape[0], dtype=np.uint8)
    neighbors_diag = np.zeros(skeleton_image_white_pixels_indices.shape[0], dtype=np.uint8)
    
    # Подсчитываем соседей для всех белых пикселей 
    for i in range(skeleton_image_white_pixels_indices.shape[0]):
        x, y = skeleton_image_white_pixels_indices[i]
        neighbors_hv[i] = np.sum(skeleton_image[x-1:x+2, y-1:y+2] * mask_hv)
        neighbors_diag[i] = np.sum(skeleton_image[x-1:x+2, y-1:y+2] * mask_diag)

    # Находим возможные координаты пересечений по превышению количества соседей
    intersections_coords = np.vstack((
        skeleton_image_white_pixels_indices[np.nonzero(neighbors_hv > 2)],
        skeleton_image_white_pixels_indices[np.nonzero(neighbors_diag > 2)]))
    
    # Находим все концы парабол
    parabolas_ends = skeleton_image_white_pixels_indices[
        np.nonzero(neighbors_hv + neighbors_diag < 2)]

    return intersections_coords

def apply_threshold_and_thinning(image_np):
    Otsu_Threshold = threshold_otsu(image_np)    
    BW_Original = image_np < Otsu_Threshold        
    BW_Skeleton = zhang_suen_thinning_optimized(BW_Original)
    return BW_Original, BW_Skeleton

def is_intersection(x, y, image):
    """检查一个像素点是否是交点，中心为黑色且周围全为白色像素"""
    if image[x, y] != 0:  # 确保中心像素是黑色
        return False
    # 获取周围的8个邻居像素
    neighbors = image[x-1:x+2, y-1:y+2]
    # 检查除了中心点之外所有邻居是否都是白色
    neighbors_flattened = neighbors.flatten()
    # 排除中心点，检查其余像素
    return np.all(neighbors_flattened[np.arange(neighbors_flattened.size) != 4] == 255)

def find_and_mark_intersections(BW_Skeleton):
    # binary_img = (BW_Skeleton * 255).astype(np.uint8)
    if BW_Skeleton.dtype != np.uint8:
        BW_Skeleton = (BW_Skeleton * 255).astype(np.uint8)
    _, binary_img = cv2.threshold(BW_Skeleton, 127, 255, cv2.THRESH_BINARY)
    intersections = []

    for x in range(1, binary_img.shape[0] - 1):
        for y in range(1, binary_img.shape[1] - 1):
            if is_intersection(x, y, binary_img):
                intersections.append((x, y))

    color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for point in intersections:
        cv2.circle(color_img, (point[1], point[0]), 1, (0, 0, 255), -1)  # Red circle marking
        print(intersections[0])
    
    return color_img, intersections

def bfs_track(image, start_points, track_mask):
    queue = deque(start_points)
    output_image = np.zeros_like(image, dtype=np.uint8)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == 255 and track_mask[nx, ny]:
                    output_image[nx, ny] = 255
                    image[nx, ny] = 0  # Mark this pixel as visited by setting it to black
                    queue.append((nx, ny))
    
    return output_image


IMAGE_FILE_NAME = "output/img3_combined_7.bmp"
img_np = cv2.imread(IMAGE_FILE_NAME, cv2.IMREAD_GRAYSCALE)
# img_np = process_and_convert_image(image_path)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.imshow('img', img_np)
cv2.waitKey()

# img_bw, img_bw_skeleton = apply_threshold_and_thinning(img_np)

ret, img_bw = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('img', img_bw)
cv2.waitKey()

# img_bw_skeleton = zhang_suen_thinning((img_bw // 255))
img_bw_skeleton = zhang_suen_thinning_optimized((img_bw // 255))

cv2.imshow('img', (img_bw_skeleton * 255).astype(np.uint8))
cv2.waitKey()

# cv2.imwrite('thinning_result.bmp', (img_bw_skeleton * 255).astype(np.uint8))
#第一种方法，老师计算的
intersections_coords = find_intersections(img_bw_skeleton)

color_img = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
color_img[img_bw_skeleton == 1] = (255, 0 , 0)
for point in intersections_coords:
    cv2.circle(color_img, (point[1], point[0]), 0, (0, 0, 255), -1)  # Red circle marking
    print(f"Intersection1 at: ({point[1]}, {point[0]})")


cv2.imshow('img', color_img[intersections_coords[0][0]-30:intersections_coords[0][0]+30,
                            intersections_coords[0][1]-30:intersections_coords[0][1]+30,:])
cv2.imwrite('transitions_Marked_3_7_1.png', color_img)
cv2.waitKey()

#第二种方法,通过计算过渡个数，找交点位置。

cv2.imwrite('thinning_result.bmp', (img_bw_skeleton * 255).astype(np.uint8))

img = cv2.imread('thinning_result.bmp', cv2.IMREAD_GRAYSCALE)
# 转换为二值图像，确保图像是0和255值
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

def transitions_count(window):
    "计算3x3窗口边界上的 (1,0,1) 过渡个数"
    boundary = np.concatenate([window[0, :], window[:, -1], window[-1, ::-1], window[::-1, 0]])
    # 将边界的值从255转换为1，0保持不变
    boundary = boundary // 255
    pattern = [1, 0, 1]
    count = 0
    for i in range(len(boundary) - len(pattern) + 1):
        if list(boundary[i:i+len(pattern)]) == pattern:
            count += 1
    return count

# 遍历图像，计算每个3x3窗口的过渡个数
rows, cols = binary_img.shape
result = np.zeros((rows - 2, cols - 2), dtype=int)

for i in range(rows - 2):
    for j in range(cols - 2):
        window = binary_img[i:i + 3, j:j + 3]
        result[i, j] = transitions_count(window)
max_val = result.max()
max_positions = np.argwhere(result == max_val)

color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 在原始图像中标记过渡次数最多的位置
for pos in max_positions:
    top_left_x, top_left_y = pos
    center_x, center_y = top_left_x + 1, top_left_y + 1  # 调整为3x3窗口的中心位置
    cv2.circle(color_img, (center_y, center_x), 1, (0, 0, 255), -1)  # 用红色圆圈标记
    print(f"Intersection2 at: ({center_y}, {center_x})")
# 保存和显示结果图像
cv2.imwrite('transitions_Marked_3_7_2.png', color_img)
cv2.imshow('Transition Count Max Marked', color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 输出或存储结果
print(result)




