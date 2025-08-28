import numpy as np
from typing import Tuple

def pixel_distance(p1: Tuple[int,int],p2:Tuple[int,int]) -> float:
    """两点之间像素距离"""
    (x1,y1),(x2,y2) = p1,p2
    return float(np.hypot(x2-x1,y2-y1))



def pixel_distance_mm(p1,p2,mm_per_pixel:float)->float:
    """像素距离转成毫米距离（根据比例尺）"""
    return pixel_distance(p1,p2)*float(mm_per_pixel)


def estimate_ball_pixel_diameter(bbox:Tuple[int,int,int,int])-> float:
    """用网球外接框来估计直径"""
    x1,y1,x2,y2= bbox
    w = max(1,x2-x1)
    h = max(1,y2-y1)
    return 0.5*(w + h)


def mm_per_pixel_from_ball(bbox:Tuple[int,int,int,int],real_ball_diameter_mm:float)->float:
    """用网球真实距离和图像像素计算比例尺"""
    d_px = estimate_ball_pixel_diameter(bbox)
    return float(real_ball_diameter_mm)/float(d_px)