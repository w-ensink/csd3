import numpy as np
import yaml


class TransformInfo:
    def __init__(self, display_points: [(int, int)], target_width: int, target_height: int):
        self.display_points = display_points
        self.target_width = target_width
        self.target_height = target_height


class Features:
    def __init__(self, center_points: [(float, float)], contours: [np.ndarray], black_white_ratio: float):
        self.center_points = center_points
        self.contours = contours
        self.black_white_ratio = black_white_ratio


Frame = np.ndarray


def parse_config(file_path: str):
    with open(file_path, 'r') as file:
        return yaml.full_load(file)