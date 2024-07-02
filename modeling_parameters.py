import random
import numpy as np


class ModelingParabolaParameters:
    def __init__(self):
        # Устанавливаем значения по умолчанию
        self.particle_diameter = 0.03 #mm
        self.x_start_trajectory = -10 #mm
        self.y_start_trajectory = 15 #mm
        self.start_speed = 0.65 * 10**3
        self.start_angle = 75
        self.plane_parameter_A = 0 
        self.plane_parameter_B = 0
        self.plane_parameter_C = 420
        self.expose_time_start = 0.0
        self.expose_time = 0.10
        self.delta_t = 0.01
        self.interval_time = 0.0001

        self.image_width = 1920
        self.image_height = 1200
        self.x_integration_step = 5*10**-3
        self.y_integration_step = 5*10**-3

        # Вектор переноса второй камеры в мм
        self.cams_trans_vec_x = 0.0
        self.cams_trans_vec_y = 0.0
        self.cams_trans_vec_z = 0.0

        # Углы поворота второй камеры в градусах
        self.cams_rot_x = 0.0
        self.cams_rot_y = 0.0
        self.cams_rot_z = 0.0

        self.cam1_K = np.array([[12900., 0., 960.], 
                                [0., 12900., 600.], 
                                [0., 0., 1.]])
        self.cam1_R = np.array([[1., 0., 0.],
                                [0, 1., 0.],
                                [0., 0., 1.]])
        self.cam1_T = np.array([0., 0., 0.])
        self.cam1_dist = np.array([0., 0., 0., 0.])
        
        self.cam2_K = np.array([[12900., 0., 960.], 
                                [0., 12900., 600.], 
                                [0., 0., 1.]])
        
        
    @property
    def cam2_R(self):
        return calculate_rotated_rotation_matrix(self.cams_rot_x, self.cams_rot_y, self.cams_rot_z)
    
    @property
    def cam2_T(self):
        return np.array([self.cams_trans_vec_x, self.cams_trans_vec_y, self.cams_trans_vec_z])
    
    @property
    def cam2_dist(self):
        return np.array([0., 0., 0., 0.])


def calculate_rotated_rotation_matrix(x: float, y: float, z: float) -> np.ndarray:
    '''
    Рассчитывает матрицу поворота по трем углам Эйлера в градусах
    '''
    # 旋转角度（弧度）,x为角度
    angle_rad_x = np.radians(x)
    angle_rad_y = np.radians(y)
    angle_rad_z = np.radians(z)

    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
                      [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]])
    
    rot_y = np.array([[np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
                      [0, 1, 0],
                      [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]])
    
    rot_z = np.array([[np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
                      [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
                      [0, 0, 1]]) 

    # 计算旋转后的相机旋转矩阵
    rotated_rotation_matrix = np.dot(np.dot(rot_x, rot_y), rot_z)

    return rotated_rotation_matrix
