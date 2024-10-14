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

        # # Приводим порядок точек единообразно слева на право
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

    # 处理末点到交点
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



def match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections, parabolas_ends, threshold=1e-2):
    # 1. 初始化有向图
    G = nx.DiGraph()

    # 2. 构建图的节点，每个片段作为一个节点
    for i in range(len(complete_parabolas)):
        G.add_node(i, parabola=complete_parabolas[i], direction=all_parabolas_directions[i])
    
    # 3. 构建图的边
    # 如果两个片段的末点和起点在交点处匹配，且方向一致，则在图中添加边
    for i in range(len(complete_parabolas)):
        end_i = parabolas_ends[i % len(parabolas_ends)]  # 从parabolas_ends获取末端点
        dir_i = all_parabolas_directions[i][-1]  # i片段的末端方向
        
        for j in range(len(complete_parabolas)):
            if i != j:
                dir_j = all_parabolas_directions[j][0]  # j片段的起点方向
                intersection_j = intersections[j % len(intersections)]  # 从intersections获取交点
                
                # 检查末端点和交点是否接近
                if np.linalg.norm(np.array(end_i) - np.array(intersection_j)) < threshold:
                    # 检查方向是否一致
                    angle_diff = np.dot(dir_i, dir_j) / (np.linalg.norm(dir_i) * np.linalg.norm(dir_j))
                    if angle_diff > 0.95:  # 角度一致性阈值
                        G.add_edge(i, j)

    # 4. 匹配抛物线片段，寻找完整路径
    # 在图中找到所有连通的子图，每个连通子图代表一个完整的抛物线
    complete_paths = []
    for subgraph in nx.weakly_connected_components(G):  # 寻找弱连通分量
        complete_paths.append(list(subgraph))
    
    # 5. 返回匹配到的两条完整的抛物线片段
    return complete_paths



#匹配片段，对于两条抛物线相交有一个交点的情况，不适应于三个或以上的抛物线
# def match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections, parabolas_ends):
#     """
#     匹配抛物线的四个部分，判断哪些片段属于同一条抛物线。
#     :param complete_parabolas: 抛物线的四个片段
#     :param all_parabolas_directions: directions of parabolas
#     :param intersection: 交点坐标
#     :return: 两对片段，分别属于两条抛物线
#     """

    # NetworkX
    # Проверяем на вхождение точки в часть параболы
    # [(par==intersections[1]).all(axis=1).any() for par in complete_parabolas]
    # Берем первую конечную точку, находим часть параболы в которую она входит, создаем вершину и одно ребро
    # Смотрим чем кончается найденная часть параболы - точкой пересечения, добавляем вершину
    # От точки пересечения смотрим какие части параболы к ней относятся, и добавляем ребра

#     matched_pairs = []
#     used_indices = set()  # 用于记录已经匹配的片段索引

#     angles_end = []
#     angles_start = []

#     for i in range(len(complete_parabolas)):
#         averaging_length = min(len(complete_parabolas[i]), 5)
#         angles_end.append(np.rad2deg(np.arctan2(*np.mean(np.array(all_parabolas_directions[i])[-averaging_length:], axis=0))))
#         angles_start.append(np.rad2deg(np.arctan2(*np.mean(np.array(all_parabolas_directions[i])[:averaging_length], axis=0))))


#     for intersection in intersections:
#         tracks_indicies_with_intersection = []

#         for i in range(len(complete_parabolas)):
#             if (intersection == complete_parabolas[i]).all(1).any():
#                 tracks_indicies_with_intersection.append(i)

#         if len(tracks_indicies_with_intersection) > 1:
#             angles_for_candidates = np.array([angles_end[i] for i in tracks_indicies_with_intersection])
#             angles_for_candidates.sort()
#             distances = np.diff(angles_for_candidates)
#             print(f'Diffs={distances}')
#             pair_id = distances.argmin()
#             matched_pairs.append((complete_parabolas[tracks_indicies_with_intersection[pair_id]], complete_parabolas[tracks_indicies_with_intersection[pair_id + 1]]))


#     # for i in range(len(complete_parabolas)):
#     #     if i in used_indices:
#     #         continue  # 如果片段已经匹配过，跳过 
#     #     for j in range(i + 1, len(complete_parabolas)):
#     #         if j in used_indices:
#     #             continue  # 如果片段已经匹配过，跳过

#     #         # 判断方向是否相反
#     #         if are_same_parabola(angles[i], angles[j]):# or are_same_parabola_1(angles[i], angles[j]):
#     #             matched_pairs.append((complete_parabolas[i], complete_parabolas[j]))  # 匹配到同一条抛物线的两个片段
#     #             used_indices.update([i, j])  # 标记这些片段已匹配
#     #             break  # 退出内层循环
#     #         else:
#     #             print(f"片段{i} 和 片段{j} 方向角度差不符合180度,跳过")

#     # if len(matched_pairs) * 2 < len(complete_parabolas):
#     #     print("存在未匹配的片段")
    
#     return matched_pairs


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
    
    # parabolas_ids_to_remove = []

    # for i in range(len(all_parabolas_directions)):
    #     if len(all_parabolas_directions[i]) < 1:
    #         parabolas_ids_to_remove.append(i)
    # for id in parabolas_ids_to_remove[-1::-1]:
    #     complete_parabolas.pop(id)
    #     all_parabolas_directions.pop(id)
 
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0), (255, 0, 0), (0, 0, 255), (255, 192, 203), (128, 0, 128), (0, 0, 128), (255, 140, 0), (169, 169, 169)]  # 绿色、黄色、紫色、青色
    for idx, parabola in enumerate(complete_parabolas):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        for point in parabola:
            cv2.circle(color_img, (point[1], point[0]), 1, color, -1)  # 使用不同的颜色标记每个片段

    # cv2.imshow('img', color_img[choose_coords[0]-40:choose_coords[0]+40,choose_coords[1]-40:choose_coords[1]+40])
    # cv2.waitKey()

    cv2.imshow('img', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_10_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

    matched = match_parabola_parts(complete_parabolas, all_parabolas_directions, intersections_coords, parabolas_ends)
    # colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0)]  
    colors = [(0, 255, 0), (255, 0, 0)]
    for idx, pair in enumerate(matched):
        color = colors[idx % len(colors)]  # 每个片段使用不同的颜色
        # for part in pair:
            # for point in part:
            #     cv2.circle(color_img, (point[1], point[0]), 5, color, -1)  # 使用不同的颜色标记每个片段
        for part_index in pair:
        # 通过索引获取实际的抛物线片段
            part = complete_parabolas[part_index]
            for point in part:  # 现在 part 是抛物线片段，可以迭代点
                cv2.circle(color_img, (point[1], point[0]), 5, color, -1)
                print(f"抛物线 {idx + 1} 已用颜色 {color} 标记。")
    
    cv2.imshow('matched parabola', color_img)
    cv2.imwrite(f'transitions_Marked/transitions_Marked_9_{os.path.basename(image_path)}', color_img)
    cv2.waitKey()

# List of image file paths
image_files = ["output/final_img1_combined_0.bmp", "output/final_img1_combined_1.bmp", "output/final_img1_combined_2.bmp", "output/final_img1_combined_3.bmp", "output/final_img1_combined_4.bmp", "output/final_img1_combined_5.bmp", "output/final_img1_combined_6.bmp", "output/final_img1_combined_7.bmp", "output/final_img1_combined_8.bmp", "output/final_img1_combined_9.bmp"]
image_files_1 = ["output_1/final_img1_combined_8.bmp", "output_1/final_img1_combined_7.bmp", "output_1/final_img1_combined_1.bmp","output_1/final_img1_combined_2.bmp", "output_1/final_img1_combined_3.bmp", "output_1/final_img1_combined_4.bmp", "output_1/final_img1_combined_6.bmp", "output_1/final_img1_combined_9.bmp"]
# Process each image
for image_path in image_files_1:
    process_image(image_path)

cv2.destroyAllWindows()





