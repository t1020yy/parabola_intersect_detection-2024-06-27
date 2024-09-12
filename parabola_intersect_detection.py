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

def filter_intersections_by_distance(intersections_coords, min_distance = 10):
    """
    过滤交点，使得任意两个交点之间的距离大于设定的最小距离。
    
    :param intersections_coords: 初始交点坐标列表
    :param min_distance: 保留交点的最小距离
    :return: 过滤后的交点坐标列表
    """
    filtered_intersections = []

    for i, point2 in enumerate(intersections_coords):
        keep_point = True
        for j, point1 in enumerate(filtered_intersections):
            dist = np.linalg.norm(np.array(point2) - np.array(point1))
            if dist < min_distance:
                keep_point = False
                break

        if keep_point:
            filtered_intersections.append(point2)

    return np.array(filtered_intersections)



# def filter_intersections_by_distance(intersections_coords, img, min_distance = 5, max_distance = 20):
#     """
#     过滤交点，使得任意两个交点之间的距离大于设定的最小距离，
#     如果两点之间的距离大于max_distance，则保留两点，
#     如果两点距离小于max_distance，遍历两点之间的像素，取中位值作为交点。
    
#     :param intersections_coords: 初始交点坐标列表
#     :param img: 输入的二值图像，用于追踪两点之间的路径
#     :param min_distance: 保留交点的最小距离
#     :param max_distance: 当两点距离大于此值时，保留两点
#     :return: 过滤后的交点坐标列表
#     """
#     filtered_intersections = list(intersections_coords)  # 拷贝列表以进行更新

#     i = 0
#     while i < len(filtered_intersections):
#         point1 = filtered_intersections[i]
#         print(f"point1 {point1}")
#         j = i + 1
#         while j < len(filtered_intersections):
#             point2 = filtered_intersections[j]
#             print(f"point2 {point2}")
#             dist = np.linalg.norm(np.array(point2) - np.array(point1))

#             if dist < min_distance:
#                 # 如果距离小于最小距离，合并为较小的点
#                 new_point = np.minimum(np.array(point2), np.array(point1))
#                 print(f"合并点 {point1} 和 {point2}，保留 {new_point}")
#                 filtered_intersections[i] = new_point  # 更新合并后的点
#                 filtered_intersections.pop(j)  # 删除 point2
#                 # 重置 j，重新与合并后的点进行距离比较
#                 j = i + 1
#             elif min_distance < dist < max_distance:
#                 # 如果距离在最小值和最大值之间，追踪路径并取中位值
#                 print(f"追踪点 {point1} 和 {point2} 之间的路径")
#                 parabalas_points_between, parabola_directions= track_parabola_until_target(point1, img, point2, 
#                                                                                            mode='between')
#                 new_point = np.median(parabalas_points_between, axis=0).astype(int)
#                 print(f"取中位值 {new_point}")
#                 filtered_intersections[i] = new_point  # 更新合并后的点为中位值
#                 filtered_intersections.pop(j)  # 删除 point2
#                 # 重置 j，重新与合并后的点进行距离比较
#                 j = i + 1
#             else:
#                 # 如果距离大于最大值，保留两个点
#                 j += 1

#         i += 1  # 处理下一个点
#         print(f"filtered_intersections {filtered_intersections}")
#     return np.array(filtered_intersections)


def label_and_filter_connected_components(img, min_size = 100): #计算连通区域，减小较少的像素区。
    """
    对二值图像进行连通组件标记，过滤掉较小的连通区域，并返回过滤后的图像。
    
    :param img: 输入的二值图像
    :param min_size: 过滤掉像素数量少于 min_size 的连通组件
    :return: 过滤后的图像
    """
    # 对二值图像进行连通组件标记
    num_labels, labels = cv2.connectedComponents(img)

    # 创建一个新图像，用于保存过滤后的连通区域
    filtered_img = np.zeros_like(img)

    # 遍历所有连通组件
    for i in range(1, num_labels):
        # 找到连通组件的像素坐标
        pts = np.where(labels == i)
        if len(pts[0]) >= min_size:  # 过滤掉较小的连通组件
            filtered_img[pts] = 255  # 保留较大的连通区域

    return filtered_img


def points_are_close(point1, point2, tolerance=1e-5):
    """
    判断两个点是否接近，允许一定的容差。
    支持 NumPy 数组和 Python 元组的比较。
    """
    if isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray):
        return np.all(np.abs(point1 - point2) < tolerance)
    else:
        return abs(point1[0] - point2[0]) < tolerance and abs(point1[1] - point2[1]) < tolerance


def track_parabola_until_target(end_point, img, target_coords, mode='intersection'):
    """
    通用的轨迹追踪函数，用于追踪末点到交点或交点到交点。
    
    :param end_point: 起始点坐标
    :param img: 二值图像
    :param target_coords: 目标点（交点或其他点）
    :param mode: 追踪模式，'intersection' 表示末点到交点，'between' 表示交点之间
    :return: 追踪到的点和方向向量
    """
    c_x, c_y = end_point
    parabalas_points = []
    parabola_directions = []
    track_directions = ((0, -1), (1, 0), (-1, 0), (0, 1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    track_directions_in_deg = np.rad2deg(np.arctan2(np.array(track_directions)[:, 0], np.array(track_directions)[:, 1]))

    while len(parabalas_points) == 0 or point_founded:
        parabalas_points.append((c_x, c_y))
        point_founded = False
        
        # 检查是否到达目标点（交点或其他点）
        if mode == 'intersection':
            if np.all(np.any(np.array([c_x, c_y]) == target_coords, axis=0)):
                print(f"Reached target intersection at: ({c_x}, {c_y})")
                break
        else:
            if np.all(np.isclose([c_x, c_y], target_coords, atol=1e-5)):
                print(f"Reached target: ({c_x}, {c_y})")
                break

        # 遍历所有方向，寻找下一个点
        for direction, direction_in_deg in zip(track_directions, track_directions_in_deg):
            t_x = c_x + direction[0]
            t_y = c_y + direction[1]
            if img[t_x, t_y] == 1 and (t_x, t_y) not in parabalas_points:
                parabola_directions.append((t_x - c_x, t_y - c_y))
                c_x = t_x
                c_y = t_y
                point_founded = True
                break

    return parabalas_points, parabola_directions


def find_complete_parabolas(filtered_intersections_coords, parabolas_ends, img, mode='until_intersection'):
    """
    追踪所有抛物线部分：包括端点到交点，以及交点到交点。
    
    :param filtered_intersections_coords: 交点列表
    :param parabolas_ends: 抛物线的端点
    :param img: 二值图像
    :param mode: 追踪模式，'until_intersection' 表示追踪末点到交点，'between' 表示追踪交点到交点
    :return: 所有的抛物线部分、方向向量和最后的点
    """
    all_parabolas = []
    all_parabolas_directions = []

    # 处理末点到交点
    if mode == 'until_intersection':
        for end in parabolas_ends:
            print(f"追踪末点到交点：{end}")
            parabola_part, parabola_directions = track_parabola_until_target(end, img, filtered_intersections_coords, 
                                                                             mode='intersection')
            all_parabolas.append(parabola_part)
            all_parabolas_directions.append(parabola_directions)

    # 处理交点到交点
    elif mode == 'between':
        for i in range(len(filtered_intersections_coords)):
            for j in range(i + 1, len(filtered_intersections_coords)):
                print(f"追踪交点 {filtered_intersections_coords[i]} 到交点 {filtered_intersections_coords[j]}")
                parabola_part, parabola_directions = track_parabola_until_target(
                    filtered_intersections_coords[i], img, filtered_intersections_coords[j], mode='between'
                )
                all_parabolas.append(parabola_part)
                all_parabolas_directions.append(parabola_directions)

    return all_parabolas, all_parabolas_directions



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
    if abs(angle_diff - 180) < 15:  # 判断方向角度和是否接近 180 度（容差10度）
        return True
    return False

def distance_condition(parabola1, parabola2, max_distance):
    """
    检查两个抛物线片段之间的距离是否在允许范围内
    :param parabola1: 第一个抛物线片段
    :param parabola2: 第二个抛物线片段
    :param max_distance: 判断是否属于同一抛物线的最大距离
    :return: 如果距离条件满足则返回 True，否则返回 False
    """
    # 获取两个抛物线片段的起点和终点
    start1, end1 = parabola1[0], parabola1[-1]
    start2, end2 = parabola2[0], parabola2[-1]

    # 判断两条抛物线的起点和终点之间的距离
    dist1 = np.linalg.norm(np.array(start1) - np.array(start2))
    dist2 = np.linalg.norm(np.array(end1) - np.array(end2))

    return dist1 < max_distance and dist2 < max_distance


def share_intersection(parabola1, parabola2, intersection_points, tolerance=5):
    """
    检查两个抛物线片段是否共享一个交点
    :param parabola1: 第一个抛物线片段
    :param parabola2: 第二个抛物线片段
    :param intersection_points: 所有的交点
    :param tolerance: 判断点重合的容差
    :return: 如果共享交点则返回 True，否则返回 False
    """
    # 获取两个抛物线的最后一个点和第一个点
    end_point1 = parabola1[-1]
    start_point2 = parabola2[0]

    # 判断它们是否共享某个交点
    for intersection in intersection_points:
        if np.linalg.norm(np.array(end_point1) - np.array(intersection)) < tolerance and \
           np.linalg.norm(np.array(start_point2) - np.array(intersection)) < tolerance:
            return True
    return False


#匹配片段，对于两条抛物线相交有一个交点的情况，不适应于三个或以上的抛物线
# def match_parabola_parts(complete_parabolas, all_parabolas_directions, intersection):
#     """
#     匹配抛物线的四个部分，判断哪些片段属于同一条抛物线。
#     :param complete_parabolas: 抛物线的四个片段
#     :param all_parabolas_directions: directions of parabolas
#     :param intersection: 交点坐标
#     :return: 两对片段，分别属于两条抛物线
#     """
#     matched_pairs = []
#     used_indices = set()  # 用于记录已经匹配的片段索引

#     angles = []

#     for i in range(len(complete_parabolas)):
#         averaging_length = min(len(complete_parabolas[i]), 30)

#         angles.append(np.rad2deg(np.arctan2(*np.mean(np.array(all_parabolas_directions[i])[-averaging_length:], axis=0))))

#     for i in range(len(complete_parabolas)):
#         if i in used_indices:
#             continue  # 如果片段已经匹配过，跳过 
#         for j in range(i + 1, len(complete_parabolas)):
#             if j in used_indices:
#                 continue  # 如果片段已经匹配过，跳过

#             # 判断方向是否相反
#             if are_same_parabola(angles[i], angles[j]):
#                 matched_pairs.append((complete_parabolas[i], complete_parabolas[j]))  # 匹配到同一条抛物线的两个片段
#                 used_indices.update([i, j])  # 标记这些片段已匹配
#                 break  # 退出内层循环
#             else:
#                 print(f"片段{i} 和 片段{j} 方向角度差不符合180度,跳过")

#     if len(matched_pairs) * 2 < len(complete_parabolas):
#         print("存在未匹配的片段")
    
#     return matched_pairs



#对于多条抛物线相交，用于多个交点，但不适应于两条一个交点的情况
def match_parabola_parts(complete_parabolas, all_parabolas_directions, intersection_points):
    """
    匹配抛物线的各个片段，判断哪些片段属于同一条抛物线。
    :param complete_parabolas: 所有抛物线的片段
    :param all_parabolas_directions: 各个抛物线片段的方向
    :param intersection_points: 所有交点的坐标
    :return: 已匹配的片段列表，属于同一抛物线的片段将被配对。
    """
    matched_pairs = []
    used_indices = set()  # 用于记录已经匹配的片段索引
    angles = []
    
    # 计算每个片段的平均方向角
    for i in range(len(complete_parabolas)):
        averaging_length = min(len(complete_parabolas[i]), 30)
        # 计算平均方向
        avg_direction = np.mean(np.array(all_parabolas_directions[i])[-averaging_length:], axis=0)
        angles.append(np.rad2deg(np.arctan2(*avg_direction)))

    # 匹配每对抛物线片段
    for i in range(len(complete_parabolas)):
        if i in used_indices:
            continue  # 如果片段已经匹配过，跳过 
        for j in range(i + 1, len(complete_parabolas)):
            if j in used_indices:
                continue  # 如果片段已经匹配过，跳过
            
            # 判断方向是否相反，且它们是否共享一个交点
            if are_same_parabola(angles[i], angles[j]) and share_intersection(complete_parabolas[i], 
                                                                              complete_parabolas[j], intersection_points):
                # 如果方向匹配并且共享一个交点，则将这两个片段匹配到同一条抛物线
                if distance_condition(complete_parabolas[i], complete_parabolas[j], max_distance = 20):
                    matched_pairs.append((complete_parabolas[i], complete_parabolas[j]))  # 匹配到同一条抛物线
                    used_indices.update([i, j])  # 标记这些片段已匹配
                    break  # 退出内层循环
                # matched_pairs.append((complete_parabolas[i], complete_parabolas[j]))  
                # used_indices.update([i, j])  # 标记这些片段已匹配
                # break  # 退出内层循环

    if len(matched_pairs) * 2 < len(complete_parabolas):
        print("存在未匹配的片段")

    return matched_pairs


def process_image(image_path):
    # Read image in grayscale
    img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Display the image
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 800, 600)
    # cv2.imshow('img', img_np)
    # cv2.waitKey()
    # Apply threshold
    ret, img_bw = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Display the binary image
    # cv2.imshow('img', img_bw)
    # cv2.waitKey()

    # 调用函数处理图像
    img_filtered = label_and_filter_connected_components(img_bw, min_size = 120)
    # cv2.imshow('Labeled Image', img_filtered)
    
    cv2.imwrite(f'transitions_Marked/transitions_Marked_6_{os.path.basename(image_path)}', img_filtered)
    cv2.waitKey(0)

    # Apply Zhang-Suen thinning algorithm
    img_bw_skeleton = zhang_suen_thinning_optimized((img_filtered // 255))
    # Display the thinned image
    # cv2.imshow('img', (img_bw_skeleton * 255).astype(np.uint8))
    # cv2.waitKey()

    # Find intersections
    intersections_coords, parabolas_ends = find_intersections(img_bw_skeleton)
    print(f"intersections_coords: {intersections_coords}")

    # filtered_intersections_coords = filter_intersections_by_distance(intersections_coords, 
    #                                                                  img_bw_skeleton, min_distance = 5, max_distance = 20)
    
    filtered_intersections_coords = filter_intersections_by_distance(intersections_coords, min_distance = 10)
    print(f"Filtered Intersections: {filtered_intersections_coords}")

    # Convert binary image to BGR for marking
    color_img = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
    color_img[img_bw_skeleton == 1] = (255, 0, 0)

    for point in intersections_coords:
        cv2.circle(color_img, (point[1], point[0]), 2, (0, 0, 255), -1)  # Red circle marking
        print(f"Intersection at: ({point[1]}, {point[0]})")

    for point in filtered_intersections_coords:
        cv2.circle(color_img, (point[1], point[0]), 3, (0, 255, 0), -1)  # 用绿色标记过滤后的交点
        print(f"交点标记在: ({point[1]}, {point[0]})")

    # first_parabola = trackParabola(parabolas_ends[0], img_bw_skeleton)
    # 找到并组合两条完整的抛物线
    
    for idx, point in enumerate(parabolas_ends):
        cv2.circle(color_img, (point[1], point[0]), 2, (0, 0, 255), -1)  # 用红色圆圈标记端点
        # 在端点旁边标注数字
        cv2.putText(color_img, str(idx + 1), (point[1] + 10, point[0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"端点 {idx + 1} 标记在: ({point[1]}, {point[0]})")

  
    # # 追踪末点到交点的轨迹
    complete_parabolas_1, all_parabolas_directions_1 = find_complete_parabolas(filtered_intersections_coords, parabolas_ends, img_bw_skeleton, mode='until_intersection')
    complete_parabolas_2, all_parabolas_directions_2 = find_complete_parabolas(filtered_intersections_coords, parabolas_ends, img_bw_skeleton, mode='between')

    # 合并所有轨迹
    complete_parabolas = complete_parabolas_1 + complete_parabolas_2
    all_parabolas_directions = all_parabolas_directions_1 + all_parabolas_directions_2


    parabolas_ids_to_remove = []

    for i in range(len(all_parabolas_directions)):
        if len(all_parabolas_directions[i]) < 1:
            parabolas_ids_to_remove.append(i)
    for id in parabolas_ids_to_remove[-1::-1]:
        complete_parabolas.pop(id)
        all_parabolas_directions.pop(id)

    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0), (255, 0, 0), (0, 0, 255),(255, 192, 203)]  # 绿色、黄色、紫色、青色
    for idx, parabola in enumerate(complete_parabolas):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for point in parabola:
            cv2.circle(color_img, (point[1], point[0]), 3, color, -1)  # 使用不同的颜色标记每个片段
    

    # cv2.imshow('img', color_img[choose_coords[0]-40:choose_coords[0]+40,choose_coords[1]-40:choose_coords[1]+40])
    # cv2.waitKey()

    cv2.imshow('img', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_10_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

    matched = match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections_coords)
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0)]  
    for idx, pair in enumerate(matched):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for part in pair:
            for point in part:
                cv2.circle(color_img, (point[1], point[0]), 8, color, -1)  # 使用不同的颜色标记每个片段
        print(f"抛物线 {idx + 1} 已用颜色 {color} 标记。")
    
    cv2.imshow('matched parabola', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_9_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

# List of image file paths
image_files = ["output/final_img1_combined_0.bmp", "output/final_img1_combined_1.bmp", "output/final_img1_combined_2.bmp", "output/final_img1_combined_3.bmp", "output/final_img1_combined_4.bmp", "output/final_img1_combined_5.bmp", "output/final_img1_combined_6.bmp", "output/final_img1_combined_7.bmp", "output/final_img1_combined_8.bmp", "output/final_img1_combined_9.bmp"]
image_files_1 = ["output_1/final_img1_combined_7.bmp", "output_1/final_img1_combined_8.bmp", "output_1/final_img1_combined_1.bmp","output_1/final_img1_combined_2.bmp", "output_1/final_img1_combined_3.bmp", "output_1/final_img1_combined_4.bmp", "output_1/final_img1_combined_6.bmp", "output_1/final_img1_combined_9.bmp"]
# Process each image
for image_path in image_files_1:
    process_image(image_path)

cv2.destroyAllWindows()





