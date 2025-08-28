from ultralytics import YOLO
import numpy as np
from typing import Iterable,List, Dict, Tuple,Optional,Generator 

class YoloDetector:
    """ 
    -detect(image):单帧检测
    -track(soucre):内置ByteTrack跟踪

    """
    # def __init__(self, model_path, target_classes=None):
    #     self.model = YOLO(model_path)
    #     self.target_classes = target_classes  # 只检测特定类别
    def __init__(self,
                 model_path:str,
                 target_classes:Optional[List[str]] = None,
                 tracker_cfg:str = "bytetrack.yaml"
                 ):
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.tracker_cfg  = tracker_cfg

        self._name_to_id = {v: k for k ,v in self.model.names.items()}
        if self.target_classes:
            self._class_ids = [self._name_to_id[n] for n in self.target_classes if n in self._name_to_id]
        else:
            self._class_ids = None
    
    def detect(self, image:np.ndarray, conf:float = 0.5,iou:float =0.4)->List[Dict]:

        results = self.model(image,conf=conf,iou=iou,classes = self._class_ids)[0] #运行置信度和IOU值设定
        detections = []
 
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name = self.model.names[cls_id]
            conf = float(box.conf.item())

            if self.target_classes is None or name in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    "name": name,
                    "class_id":cls_id,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy)
                })

        return detections


    def track(
            self,
            source:Iterable | str |int,
            conf: float = 0.5,
            iou :float = 0.4,
            imgsz: int = 640,
            device :Optional[str] = None,
            save:bool = False
    ) -> Generator[Dict[str,object],None,None]:
        """
        -source:文件路径/文件夹/摄像头索引
        -save:是否保存可视化视频/结果到runs
        """
