import os
from PIL import Image
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



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
    intersections_coords = skeleton_image_white_pixels_indices[np.nonzero(neighbors_hv +  neighbors_diag > 4)]
        # np.vstack((
        # skeleton_image_white_pixels_indices[np.nonzero(neighbors_hv > 2)],
        # skeleton_image_white_pixels_indices[np.nonzero(neighbors_diag > 2)]))
    
    # Находим все концы парабол
    parabolas_ends = skeleton_image_white_pixels_indices[
        np.nonzero(neighbors_hv + neighbors_diag < 2)]

    return intersections_coords, parabolas_ends


def find_intersections_with_morphologic(skeleton_image):
    skeleton_image_gray = skeleton_image * 255

    kernels = []

    kernels.append(np.array([[ 0,  0,  0,  0,  0],
                             [ 0,  0, -1,  0,  0],
                             [ 0,  1,  1,  1,  0],
                             [ 0, -1,  1, -1,  0],
                             [-1,  0,  0,  0, -1]], np.int8))
    
    kernels.append(np.rot90(kernels[0], 1))
    kernels.append(np.rot90(kernels[0], 2))
    kernels.append(np.rot90(kernels[0], 3))

    kernels.append(np.array([[-1,  1, -1],
                             [-1,  1,  1],
                             [ 1, -1, -1]], np.int8))
    
    kernels.append(np.rot90(kernels[4], 1))
    kernels.append(np.rot90(kernels[4], 2))
    kernels.append(np.rot90(kernels[4], 3))
    
    hitormiss_intersections = np.zeros(skeleton_image.shape, dtype=np.uint8)

    for kernel in kernels:
        hitormiss = cv2.morphologyEx(skeleton_image_gray, cv2.MORPH_HITMISS, kernel)    
        hitormiss_intersections = cv2.bitwise_or(hitormiss, hitormiss_intersections)
   
    intersections_coords = np.nonzero(hitormiss_intersections == 255)

    intersections_coords = tuple(np.array([x, y]) for x, y in zip(*intersections_coords))

    kernels = []

    kernels.append(np.array([[-1,  1, -1],
                             [-1,  1, -1],
                             [-1, -1, -1]], np.int8))
    
    kernels.append(np.rot90(kernels[0], 1))
    kernels.append(np.rot90(kernels[0], 2))
    kernels.append(np.rot90(kernels[0], 3))

    kernels.append(np.array([[-1, -1,  1],
                             [-1,  1, -1],
                             [-1, -1, -1]], np.int8))
    
    kernels.append(np.rot90(kernels[4], 1))
    kernels.append(np.rot90(kernels[4], 2))
    kernels.append(np.rot90(kernels[4], 3))
    
    hitormiss_tracks_end = np.zeros(skeleton_image.shape, dtype=np.uint8)

    for kernel in kernels:
        hitormiss = cv2.morphologyEx(skeleton_image_gray, cv2.MORPH_HITMISS, kernel)    
        hitormiss_tracks_end = cv2.bitwise_or(hitormiss, hitormiss_tracks_end)

    parabolas_ends = np.nonzero(hitormiss_tracks_end == 255)
    parabolas_ends = tuple(np.array([x, y]) for x, y in zip(*parabolas_ends))

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


def label_and_filter_connected_components(img, min_size = 150): #计算连通区域，减小较少的像素区。
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


def track_parabola_until_target(start_point, img, target_point, points_used):
    """
    通用的轨迹追踪函数，用于追踪末点到交点或交点到交点。
    
    :param start_point: 起始点坐标
    :param img: 二值图像
    :param target_point: 目标点（交点或其他点）
    :return: 追踪到的点和方向向量
    """
    c_x, c_y = start_point
    parabalas_points = []
    parabola_directions = []
    track_directions = ((0, -1), (1, 0), (-1, 0), (0, 1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    # track_directions_in_deg = np.rad2deg(np.arctan2(np.array(track_directions)[:, 0], np.array(track_directions)[:, 1]))

    first_point = True
    wait_key_period = 1

    previous_direction = (0, 0)

    while len(parabalas_points) == 0 or point_founded:
        parabalas_points.append((c_x, c_y))
        points_used[c_x, c_y] = True
        point_founded = False
    
        # # Display tracking
        # color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # color_img[img == 1] = (255, 0, 0)
        # color_img[points_used] = (255, 255, 255)
        # color_img[target_point[:,0], target_point[:,1], :] = (0, 0, 255)
        # cv2.drawMarker(color_img, (c_y, c_x), (0, 255, 0), cv2.MARKER_SQUARE, 5)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # 使用WINDOW_NORMAL，允许调整窗口大小

        # # 设置窗口的尺寸，例如宽 800 像素，高 600 像素
        # cv2.resizeWindow('img', 1700, 1200)
        # cv2.imshow('img', color_img)
        # k = cv2.waitKey(wait_key_period)

        # if k == 27:
        #     wait_key_period = 1
        
        if not first_point and np.any(np.all(np.array([c_x, c_y]) == target_point, axis=1)):
            print(f"Reached target intersection at: ({c_x}, {c_y})")
            # Убираем точку из использованных, чтобы можно было попасть на конечную точку при проходе повторно
            points_used[c_x, c_y] = False
            point_founded = True
            break

        first_point = False

        # 遍历所有方向，寻找下一个点
        for direction in track_directions:
            t_x = c_x + direction[0]
            t_y = c_y + direction[1]
            # if img[t_x, t_y] == 1 and (direction[0] != -previous_direction[0] and direction[1] != -previous_direction[1]) and not points_used[t_x, t_y]:
            if img[t_x, t_y] == 1 and not points_used[t_x, t_y]:
                parabola_directions.append((t_x - c_x, t_y - c_y))
                previous_direction = direction
                c_x = t_x
                c_y = t_y
                point_founded = True                
                break

    # Приводим порядок точек единообразно слева на право
    if parabalas_points[0][0] > parabalas_points[-1][0]:
        parabalas_points.reverse()
        parabola_directions.reverse()
        parabola_directions = [[x * -1, y * -1] for x, y in parabola_directions]

    return parabalas_points, parabola_directions, point_founded


def find_complete_parabolas(intersections_coords, parabolas_ends, img):
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

    parabolas_ends_and_intersections = np.vstack((parabolas_ends, intersections_coords))

    points_used = np.zeros(img.shape, dtype=np.bool_)

    for intersection in intersections_coords:
        print(f"追踪末点到交点：{intersection}")

        point_founded = True
        while point_founded:
            parabola_part, parabola_directions, point_founded = track_parabola_until_target(intersection, img, parabolas_ends_and_intersections, points_used) 
            if not point_founded:
                break
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

def check_parabolas_endpoints(complete_parabolas):
    """打印所有抛物线片段的起点和终点的坐标"""
    for i, parabola in enumerate(complete_parabolas):
        parabola = np.array(parabola)  # 确保转换为 numpy 数组
        start_point = parabola[0]  # 片段的起点
        end_point = parabola[-1]   # 片段的终点
        distance = np.linalg.norm(end_point - start_point)
        print(f"Parabola {i}: Start Point = {start_point}, End Point = {end_point}, Distance = {distance}")
   
# best_parabola, _ = find_best_parabola(next_start_point, candidate_parabolas, connected_parabolas[-2])
def find_best_parabola(next_start_point, candidate_parabolas, current_parabola):
    """找到与 current_tracking_point 相连的最佳片段，基于残差差异选择."""
    best_residuals = 100000
    best_parabola = None
    for candidate in candidate_parabolas:

        candidate_residuals = calculate_partial_residuals(current_parabola, candidate, next_start_point)

        if candidate_residuals < best_residuals:
            best_parabola = candidate
            best_residuals = candidate_residuals
    
    return best_parabola, best_residuals

def find_matching_parabolas(start_point, current_parabola, parabolas):
    matching_parabolas = []
    for parabola in parabolas:
        # 将 parabola 转换为 numpy 数组，确保数据类型正确
        parabola = np.array(parabola)

        if np.array_equal(parabola, current_parabola):
            continue
        
        if np.array_equal(parabola[0], start_point) or np.array_equal(parabola[-1], start_point):
            matching_parabolas.append(parabola)
           
    return matching_parabolas

def calculate_partial_residuals(parabola1, parabola2, common_point, n_points = 50):
    """
    计算给定抛物线片段的前 n_points 个点和后 n_points 个点的残差
    """
    parabolas_ends = []
    for parabola in [parabola1, parabola2]:
        # 确保片段转换为 NumPy 数组
        parabola = np.array(parabola)
        
        # 前 n_points 个点
        if len(parabola) < n_points:
            parabolas_ends.append(parabola[:])
            # x_vals, y_vals = parabola[:, 0], parabola[:, 1]
        else:
            if np.all(np.isclose(parabola[0], common_point, atol = 30)):
                parabolas_ends.append(parabola[:n_points])
                # x_vals, y_vals = parabola[:n_points, 0], parabola[:n_points, 1]
            else:
                parabolas_ends.append(parabola[-n_points:])
                # x_vals, y_vals = parabola[-n_points:, 0], parabola[-n_points:, 1]

    x_vals = np.vstack(parabolas_ends)[:,0]
    y_vals = np.vstack(parabolas_ends)[:,1]
    
    coef, residuals, _, _, _ = np.polyfit(x_vals, y_vals, 2, full=True)

    return residuals
        

def trace_parabolas(start_point, complete_parabolas):
    connected_parabolas = []  # 每次追踪到的片段
    current_parabola = None
    # previous_residuals = 0
    processed_points = set() 

    for parabola in complete_parabolas:
        if np.isclose(parabola[0], start_point).all() or np.isclose(parabola[-1], start_point).all():
            current_parabola = parabola
            connected_parabolas.append(current_parabola)  # 添加第一个片段
            break
        
    if current_parabola is None:
        return connected_parabolas  # 如果没有匹配的片段，返回
        
    # 将当前点标记为已处理
    processed_points.add(tuple(start_point))

    current_parabola = np.array(current_parabola)

    if np.all(np.isclose(current_parabola[-1], start_point)):
        # residuals, _ = calculate_partial_residuals(current_parabola)
        next_start_point = current_parabola[0]
    else:
        # _, residuals = calculate_partial_residuals(current_parabola)
        next_start_point = current_parabola[-1]
    # previous_residuals = residuals

    while next_start_point is not None:

        processed_points.add(tuple(next_start_point))

        candidate_parabolas = find_matching_parabolas(next_start_point, current_parabola, complete_parabolas)

        # 如果没有候选片段，终止追踪
        if not candidate_parabolas:
            break

        if len(current_parabola) < 20 and len(connected_parabolas) > 1:
            best_parabola, _ = find_best_parabola(next_start_point, candidate_parabolas, connected_parabolas[-2])
        else:
            best_parabola, _ = find_best_parabola(next_start_point, candidate_parabolas, current_parabola)
        if best_parabola is None:
            break
            
        connected_parabolas.append(best_parabola)
        current_parabola = best_parabola

        # 更新 next_start_point 为新片段的另一端，继续追踪
        if np.all(np.isclose(current_parabola[-1], next_start_point)):
            next_start_point = current_parabola[0]
        else:
            next_start_point = current_parabola[-1]   
        # previous_residuals = best_residuals

    return connected_parabolas   

def match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections, parabolas_ends, color_img):
    """匹配抛物线片段，并根据拟合的系数和残差找到相同抛物线的段。"""

    matched_pairs = []  # 用于保存所有起始点追踪到的片段链
    processed_points = set()
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0), (255, 0, 0), (0, 0, 255), (255, 192, 203), (128, 0, 128), (0, 0, 128), (255, 140, 0), (169, 169, 169)]
    color_index = 0 

    for end_point in parabolas_ends:
        # 跳过已处理的 end_point
        if tuple(end_point) in processed_points:
            continue
        connected_parabolas = trace_parabolas(end_point, complete_parabolas)


        color = colors[color_index % len(colors)]
        color_index += 1  
        for parabola in connected_parabolas:
            for point in parabola:
                cv2.circle(color_img, (point[1], point[0]), 3, color, -1)  # 使用不同的颜色标记每个片段
        # cv2.imshow('img', color_img)
        # cv2.waitKey()

        if connected_parabolas:
            matched_pairs.append(connected_parabolas)

    return matched_pairs

def process_image(image_path):
    # Read image in grayscale
    img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Display the image
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 800)
    cv2.namedWindow('matched parabola', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('matched parabola', 1000, 800)
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
    # cv2.waitKey(0)

    # Apply Zhang-Suen thinning algorithm
    img_bw_skeleton = zhang_suen_thinning_optimized((img_filtered // 255))
    # Display the thinned image
    # cv2.imshow('img', (img_bw_skeleton * 255).astype(np.uint8))
    # cv2.waitKey()

    # Find intersections
    # intersections_coords, parabolas_ends = find_intersections(img_bw_skeleton)
    intersections_coords, parabolas_ends = find_intersections_with_morphologic(img_bw_skeleton)
    print(f"intersections_coords: {intersections_coords}")

    # Convert binary image to BGR for marking
    color_img = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
    color_img[img_bw == 255] = (125, 125, 125)
    color_img[img_bw_skeleton == 1] = (255, 255, 255)

    for point in intersections_coords:
        cv2.circle(color_img, (point[1], point[0]), 0, (0, 0, 255), -1)  # 用绿色标记过滤后的交点
        print(f"交点标记在: ({point[1]}, {point[0]})")
          
    for idx, point in enumerate(parabolas_ends):
        cv2.circle(color_img, (point[1], point[0]), 0, (0, 255, 255), -1)  # 用红色圆圈标记端点
        # 在端点旁边标注数字
        cv2.putText(color_img, str(idx + 1), (point[1] + 10, point[0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"端点 {idx + 1} 标记在: ({point[1]}, {point[0]})")

    cv2.namedWindow('intersection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('intersection', 800, 600)

    for point in intersections_coords:
        cropped_img = color_img[point[0] - 25:point[0] + 25, point[1] - 25:point[1] + 25]
        resized_img = cv2.resize(cropped_img, (800, 600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('intersection', resized_img)
        cv2.waitKey()

    
    # cv2.destroyWindow('intersection')
  
    # # 追踪末点到交点的轨迹
    complete_parabolas, all_parabolas_directions = find_complete_parabolas(intersections_coords, parabolas_ends, img_bw_skeleton)
    check_parabolas_endpoints(complete_parabolas)
    # parabolas_ids_to_remove = []

    # for i in range(len(all_parabolas_directions)):
    #     if len(all_parabolas_directions[i]) < 1:
    #         parabolas_ids_to_remove.append(i)
    # for id in parabolas_ids_to_remove[-1::-1]:
    #     complete_parabolas.pop(id)
    #     all_parabolas_directions.pop(id)
 
    colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 165, 0), (255, 255, 0), (0, 0, 255), (255, 192, 203), (128, 0, 128), (0, 0, 128), (255, 140, 0), (169, 169, 169)]  # 绿色、黄色、紫色、青色
    for idx, parabola in enumerate(complete_parabolas):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for point in parabola:
            cv2.circle(color_img, (point[1], point[0]), 1, color, -1)  # 使用不同的颜色标记每个片段

    # cv2.imshow('img', color_img[choose_coords[0]-40:choose_coords[0]+40,choose_coords[1]-40:choose_coords[1]+40])
    # cv2.waitKey()

    cv2.imshow('img', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_10_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()
      
    matched = match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections_coords, parabolas_ends, color_img)[:2]
   
    for idx, pair in enumerate(matched):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for part in pair:
            for point in part:
                cv2.circle(color_img, (point[1], point[0]), 5, color, -1)  # 使用不同的颜色标记每个片段
        # for part_index in pair:
        # # 通过索引获取实际的抛物线片段
        #     part = complete_parabolas[part_index]
        #     for point in part:  # 现在 part 是抛物线片段，可以迭代点
        #         cv2.circle(color_img, (point[1], point[0]), 5, color, -1)
        #         print(f"抛物线 {idx + 1} 已用颜色 {color} 标记。")
    
    cv2.imshow('matched parabola', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_9_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

# List of image file paths
image_files = ["output/final_img1_combined_0.bmp", "output/final_img1_combined_1.bmp", "output/final_img1_combined_2.bmp", "output/final_img1_combined_3.bmp", "output/final_img1_combined_4.bmp", "output/final_img1_combined_5.bmp", "output/final_img1_combined_6.bmp", "output/final_img1_combined_7.bmp", "output/final_img1_combined_8.bmp", "output/final_img1_combined_9.bmp"]
image_files_1 = ["output_1/final_img1_combined_8.bmp", "output_1/final_img1_combined_7.bmp","output_1/final_img1_combined_2.bmp", "output_1/final_img1_combined_4.bmp",  "output_1/final_img1_combined_6.bmp", "output_1/final_img1_combined_9.bmp", "output_1/final_img1_combined_1.bmp", "output_1/final_img1_combined_3.bmp","output_1/final_img1_combined_0.bmp"]

# Process each image
for image_path in image_files_1:
    process_image(image_path)

cv2.destroyAllWindows()





