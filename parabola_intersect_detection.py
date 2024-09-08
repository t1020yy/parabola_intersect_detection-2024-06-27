import os
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

    return intersections_coords, parabolas_ends

# def trackParabola(end_point, img):
#     c_x, c_y = end_point[0], end_point[1]
    
#     parabalas_points = []
#     parabola_directions = []
#     track_directions = ((0, -1), (1, 0), (-1, 0), (0, 1), (1, -1), (1, 1), (-1, 1), (-1, -1))
#     track_directions_in_deg = np.rad2deg(np.arctan2(np.array(track_directions)[:,0],np.array(track_directions)[:,1]))

#     while len(parabalas_points) == 0 or point_founded:
#         parabalas_points.append((c_x, c_y))
#         point_founded = False
       
#         for direction, direction_in_deg in zip(track_directions, track_directions_in_deg):
#             t_x = c_x + direction[0]
#             t_y = c_y + direction[1]
#             if img[t_x, t_y] == 1 and (t_x, t_y) not in parabalas_points:
#                 if len(parabola_directions) > 10:
#                     dir = np.mean(np.array(parabola_directions[-10:]), axis=0)
#                     dir_in_deg = np.rad2deg(np.arctan2(dir[0], dir[1]))
#                     if abs(dir_in_deg - direction_in_deg) > 45:
#                         print(f'{dir_in_deg} - {direction_in_deg}')
                    
#                 parabola_directions.append((t_x-c_x, t_y-c_y))
#                 c_x = t_x
#                 c_y = t_y
#                 point_founded = True
#                 break

#     return parabalas_points


def points_are_close(point1, point2, tolerance=1e-5):
    """
    判断两个点是否接近，允许一定的容差。
    支持 NumPy 数组和 Python 元组的比较。
    """
    if isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray):
        return np.all(np.abs(point1 - point2) < tolerance)
    else:
        return abs(point1[0] - point2[0]) < tolerance and abs(point1[1] - point2[1]) < tolerance

def trackParabola_until_intersection(end_point, img, choose_coords):
    c_x, c_y = end_point[0], end_point[1]
    
    print(f"choose_coords",choose_coords)
    parabalas_points = []
    parabola_directions = []
    track_directions = ((0, -1), (1, 0), (-1, 0), (0, 1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    track_directions_in_deg = np.rad2deg(np.arctan2(np.array(track_directions)[:,0], np.array(track_directions)[:,1]))

    while len(parabalas_points) == 0 or point_founded:
        parabalas_points.append((c_x, c_y))
        point_founded = False
        
        # 检查是否到达交叉点
        if points_are_close(np.array([c_x, c_y]), np.array(choose_coords)):
            print(f"Reached intersection at: ({c_x}, {c_y})")
            break
        last_point = (c_x, c_y)
        for direction, direction_in_deg in zip(track_directions, track_directions_in_deg):
            t_x = c_x + direction[0]
            t_y = c_y + direction[1]
            if img[t_x, t_y] == 1 and (t_x, t_y) not in parabalas_points:
                parabola_directions.append((t_x-c_x, t_y-c_y))
                c_x = t_x
                c_y = t_y
                point_founded = True
                break

        if not point_founded:
            break  # 没有找到符合条件的下一个点，退出

    return parabalas_points, last_point

def find_complete_parabolas(choose_coords, parabolas_ends, img):
    all_parabolas = []
    all_last_point = []
    for end in parabolas_ends:
        print(f"parabolas_ends1",end)
        # 从每个端点追踪到交点
        parabola_part, last_point = trackParabola_until_intersection(end, img, [choose_coords])
        all_parabolas.append(parabola_part)
        all_last_point.append(last_point)

    return all_parabolas, all_last_point

def calculate_direction(point1, point2):
    """
    计算从 point1 到 point2 的方向角度（以度为单位）。
    """
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle = np.rad2deg(np.arctan2(delta_y, delta_x))  # 计算方向角度
    print(f"Point1: {point1}, Point2: {point2}, Angle: {angle}")
    return angle

def are_same_parabola(direction1, direction2):
    """
    判断两个方向是否相反，且方向角度之和接近 180 度。
    """
    angle_diff = abs(direction1 - direction2) % 360  # 计算角度差
    print(f"Comparing directions: {direction1} vs {direction2}, angle_diff = {angle_diff}")
    if abs(angle_diff - 180) < 10:  # 判断方向角度和是否接近 180 度（容差10度）
        return True
    return False

def match_parabola_parts(complete_parabolas, intersection, all_last_point):
    """
    匹配抛物线的四个部分，判断哪些片段属于同一条抛物线。
    :param complete_parabolas: 抛物线的四个片段
    :param intersection: 交点坐标
    :return: 两对片段，分别属于两条抛物线
    """
    matched_pairs = []
    used_indices = set()  # 用于记录已经匹配的片段索引
    
    for i in range(len(complete_parabolas)):
        if i in used_indices:
            continue  # 如果片段已经匹配过，跳过
        for j in range(i + 1, len(complete_parabolas)):
            if j in used_indices:
                continue  # 如果片段已经匹配过，跳过
            
            part1 = all_last_point[i]
            part2 = all_last_point[j]
            
            # 分别计算两个片段在交点处的方向角度
            direction1 = calculate_direction(intersection, part1)  # part1 从交点开始的方向
            direction2 = calculate_direction(intersection, part2)  # part2 从交点开始的方向

            # 判断方向是否相反
            if are_same_parabola(direction1, direction2):
                matched_pairs.append((complete_parabolas[i], complete_parabolas[j]))  # 匹配到同一条抛物线的两个片段
                used_indices.update([i, j])  # 标记这些片段已匹配
                break  # 退出内层循环
            else:
                print(f"片段{i} 和 片段{j} 方向角度差不符合180度,跳过")

    if len(matched_pairs) * 2 < len(complete_parabolas):
        print("存在未匹配的片段")
    
    return matched_pairs

def process_image(image_path):
    # Read image in grayscale
    img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Display the image
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 800, 600)
    cv2.imshow('img', img_np)
    cv2.waitKey()
    # Apply threshold
    ret, img_bw = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Display the binary image
    cv2.imshow('img', img_bw)
    cv2.waitKey()

    # Apply Zhang-Suen thinning algorithm
    img_bw_skeleton = zhang_suen_thinning_optimized((img_bw // 255))
    # Display the thinned image
    cv2.imshow('img', (img_bw_skeleton * 255).astype(np.uint8))
    cv2.waitKey()

    # Find intersections
    intersections_coords, parabolas_ends = find_intersections(img_bw_skeleton)
    # Convert binary image to BGR for marking
    color_img = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
    color_img[img_bw_skeleton == 1] = (255, 0, 0)

    # first_parabola = trackParabola(parabolas_ends[0], img_bw_skeleton)
    # 找到并组合两条完整的抛物线

    for idx, point in enumerate(parabolas_ends):
        cv2.circle(color_img, (point[1], point[0]), 5, (0, 0, 255), -1)  # 用红色圆圈标记端点
        # 在端点旁边标注数字
        cv2.putText(color_img, str(idx + 1), (point[1] + 10, point[0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"端点 {idx + 1} 标记在: ({point[1]}, {point[0]})")

    choose_coords = intersections_coords[1]
    complete_parabolas, all_last_point = find_complete_parabolas(choose_coords, parabolas_ends, img_bw_skeleton)
    print("all_last_point", all_last_point)
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0)]  # 绿色、黄色、紫色、青色
    for idx, parabola in enumerate(complete_parabolas):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for point in parabola:
            cv2.circle(color_img, (point[1], point[0]), 1, color, -1)  # 使用不同的颜色标记每个片段
              
    for point in intersections_coords:
        cv2.circle(color_img, (point[1], point[0]), 0, (0, 0, 255), -1)  # Red circle marking
        print(f"Intersection at: ({point[1]}, {point[0]})")

    cv2.imshow('img', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_3_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

    matched = match_parabola_parts(complete_parabolas, choose_coords, all_last_point)
    colors = [(255, 0, 0), (0, 0, 0)]  
    for idx, pair in enumerate(matched):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for part in pair:
            for point in part:
                cv2.circle(color_img, (point[1], point[0]), 1, color, -1)  # 使用不同的颜色标记每个片段
        print(f"抛物线 {idx + 1} 已用颜色 {color} 标记。")
    
    cv2.imshow('matched parabola', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_4_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

    # for point in first_parabola:
    #     cv2.circle(color_img, (point[1], point[0]), 0, (0, 255, 0), -1)  # Green circle marking
    #     #print(f"End point at: ({point[1]}, {point[0]})")

    # 绘制两条完整的抛物线
    # for parabola in complete_parabolas:
    #     for point in parabola:
    #         cv2.circle(color_img, (point[1], point[0]), 0, (0, 255, 0), -1)  # 绿色标记完整抛物线
    #         #print(f"Point at: ({point[1]}, {point[0]})")


    # Display the marked image
    # cv2.imshow('img', color_img)
    # cv2.imwrite(f'transitions_Marked/transitions_Marked_3_{os.path.basename(image_path)}', color_img)
    # cv2.waitKey()

# List of image file paths
image_files = ["output/final_img1_combined_0.bmp", "output/final_img1_combined_1.bmp", "output/final_img1_combined_2.bmp", "output/final_img1_combined_3.bmp", "output/final_img1_combined_4.bmp", "output/final_img1_combined_5.bmp", "output/final_img1_combined_6.bmp", "output/final_img1_combined_7.bmp", "output/final_img1_combined_8.bmp", "output/final_img1_combined_9.bmp"]
image_files_1 = ["output_1/final_img1_combined_1.bmp","output_1/final_img1_combined_2.bmp", "output_1/final_img1_combined_3.bmp", "output_1/final_img1_combined_4.bmp", "output_1/final_img1_combined_6.bmp", "output_1/final_img1_combined_7.bmp", "output_1/final_img1_combined_8.bmp", "output_1/final_img1_combined_9.bmp"]
# Process each image
for image_path in image_files_1:
    process_image(image_path)

cv2.destroyAllWindows()


# #第二种方法,通过计算过渡个数，找交点位置。

# cv2.imwrite('thinning_result.bmp', (img_bw_skeleton * 255).astype(np.uint8))

# img = cv2.imread('thinning_result.bmp', cv2.IMREAD_GRAYSCALE)
# # 转换为二值图像，确保图像是0和255值
# _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# def transitions_count(window):
#     "计算3x3窗口边界上的 (1,0,1) 过渡个数"
#     boundary = np.concatenate([window[0, :], window[:, -1], window[-1, ::-1], window[::-1, 0]])
#     # 将边界的值从255转换为1，0保持不变
#     boundary = boundary // 255
#     pattern = [1, 0, 1]
#     count = 0
#     for i in range(len(boundary) - len(pattern) + 1):
#         if list(boundary[i:i+len(pattern)]) == pattern:
#             count += 1
#     return count

# # 遍历图像，计算每个3x3窗口的过渡个数
# rows, cols = binary_img.shape
# result = np.zeros((rows - 2, cols - 2), dtype=int)

# for i in range(rows - 2):
#     for j in range(cols - 2):
#         window = binary_img[i:i + 3, j:j + 3]
#         result[i, j] = transitions_count(window)
# max_val = result.max()
# max_positions = np.argwhere(result == max_val)

# color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # 在原始图像中标记过渡次数最多的位置
# for pos in max_positions:
#     top_left_x, top_left_y = pos
#     center_x, center_y = top_left_x + 1, top_left_y + 1  # 调整为3x3窗口的中心位置
#     cv2.circle(color_img, (center_y, center_x), 1, (0, 0, 255), -1)  # 用红色圆圈标记
#     print(f"Intersection2 at: ({center_y}, {center_x})")
# # 保存和显示结果图像
# cv2.imwrite('transitions_Marked_3_7_2.png', color_img)
# cv2.imshow('Transition Count Max Marked', color_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 输出或存储结果
# print(result)




