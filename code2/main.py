import cv2
import numpy as np
from detector import YoloDetector
from distance_utils import pixel_distance_mm,mm_per_pixel_from_ball
from action import send_control_command
import time


# 各项参数
MODEL_PATH = r"E:/gemini335python/335/bestcar4.pt"   # 模型路径
TARGET_CLASSES = ["car", "tennis"]   # 想跟踪的类别，全类可改成None
CONF_THRES = 0.42
IOU_THRES = 0.5
DIST_MM_THRESHOLD = 200              # 距离阈值(mm)
TRACKER_CFG = "bytetrack.yaml"       # 用 Ultralytics 自带的配置

LOST_MAX_FRAMES = 20                 #循序连续丢失的最大帧数
VIS_FILTER_ONLY_TARGETS = True      #仅计算锁定ID的那一对目标
TENNIS_DIAMETER_MM = 67.0
EMA_ALPHA = 1.0  



CAP_SOURCE = 0                     #相机索引
# def to_bgr(img):
#     """将 RGBA 转成 BGR；如果已是 BGR 就原样返回。"""
#     if img is None:
#         return None
#     if img.ndim == 3 and img.shape[2] == 4:
#         return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     return img


def main():
    #接受相机
    cap = cv2.VideoCapture(CAP_SOURCE)
    if not cap.isOpened():
        return RuntimeError(f"无法打开视频源{CAP_SOURCE}")

    #运行YOLO
    yolo = YoloDetector(MODEL_PATH, target_classes=TARGET_CLASSES)

    #锁定id状态
    was_moving_forward = False
    target_tennis_id = None
    target_car_id = None
    grabbing = False

    lost_tennis_frames = 0
    lost_car_frames = 0

    mm_per_pixel = None
    calibrated = False

    # while True:
    #     # 取彩色和深度帧
    #     ok,frame = cap.read()
    #     if not ok:
    #         print("WARN: returned None, skip this frame")
    #         continue  # 丢帧时跳过但不退出


    #     if frame.ndim == 3 and frame.shape[2] == 4:
    #         frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)



    #用内置 ByteTrack 做“检测+跟踪”（persist=True可跨帧维持ID）
    stream = yolo.model.track(
        source=CAP_SOURCE,                  # 直接传单帧 numpy 图像
        conf=CONF_THRES,
        iou=IOU_THRES,
        tracker=TRACKER_CFG,              # bytetrack配置文件
        persist=True,
        stream= True,                     # 保持上一帧轨迹状态
        verbose=False
    )
    for r in  stream:
        if getattr(r,"orig_img",None) is not None:
            frame = r.orig_img.copy()
        else:
            frame = r.plot()

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        # # 没有检测到目标，刷新画面继续
        # if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        #     # cv2.imshow("Color Image with Tracking", color_bgr)
        #     # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     #     break
        #     continue

        # boxes = results[0].boxes  # Boxes：含 .xyxy/.conf/.cls/.id
        car_detections, tennis_detections = [], []

        #可视化 + 分类收集
        for b in boxes:
            # 兼容张量/标量
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item() if hasattr(b.conf[0], "item") else b.conf[0])
            cls_id = int(b.cls[0].item() if hasattr(b.cls[0], "item") else b.cls[0])
            tid = int(b.id[0].item()) if (hasattr(b, "id") and b.id is not None) else -1
            name = yolo.model.names.get(cls_id, str(cls_id))

            # 类别过滤（如果在 detector 里没过滤，这里再保底一次）
            if TARGET_CLASSES is not None and name not in TARGET_CLASSES:
                continue

            
            # 汇总本帧目标
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            det = {"name": name, "conf": conf, "bbox": (x1, y1, x2, y2), "center": (cx, cy), "id": tid}
            if name == "car":
                car_detections.append(det)
            elif name == "tennis":
                tennis_detections.append(det)
            #渲染控制：未锁定正常画框，已锁定按是否锁定id决定画法

            if(target_tennis_id is not None )and (target_car_id is not None):
                
                is_target = (name == "tennis"and tid == target_tennis_id)or(name == "car" and tid == target_car_id)
                if VIS_FILTER_ONLY_TARGETS and not is_target:
                    continue  #仅显示锁定目标，其他不画
                color = (0,200,0)if is_target else (160,160,160)
            else:
                color = (0,200,0) #未锁定，all要画


            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{name}#{tid}",(x1,max(0,y1 -8)),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        #如果没计算比例尺，进行计算
        if not calibrated and tennis_detections:
            ball_det = max(tennis_detections,key = lambda d: (d["bbox"][2] - d["bbox"][0] )*(d["bbox"][3] -d["bbox"][1]) )
            est = mm_per_pixel_from_ball(ball_det["bbox"],TENNIS_DIAMETER_MM)
            mm_per_pixel = est if mm_per_pixel is None else(EMA_ALPHA *est + (1.0 - EMA_ALPHA)* mm_per_pixel)
            calibrated = True if EMA_ALPHA >= 1.0 else False
            cv2.putText(frame,f"CAlibrated mm/px = {mm_per_pixel:.3f}",
                        (20,80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0) ,2)
            print(f"calibrated:{mm_per_pixel:.6f}")
        



        if(target_tennis_id is None)and tennis_detections and car_detections:
            first_ball = tennis_detections[0]
            target_tennis_id = first_ball["id"]

            bx,by = first_ball["center"]
            nearest_car = min(car_detections, key=lambda c: (c["center"][0]-bx)**2 +(c["center"][1] - by)**2)
            target_car_id =nearest_car["id"]

            lost_tennis_frames = 0
            lost_car_frames = 0
            print(f"LOCK仅计算该ID{target_tennis_id}和{target_car_id}")

        #只对锁定ID计算距离和控制
        if(mm_per_pixel is not None) and (target_tennis_id is not None)and (target_car_id is not None)and not grabbing:
            #当前帧查找锁定ID
            ball = next((d for d in tennis_detections if d ["id"] == target_tennis_id),None)
            car = next((d for d in car_detections if d ["id"] == target_car_id),None)
            #丢失计数（三元运算符）
            lost_tennis_frames = lost_tennis_frames+1 if ball is None else 0
            lost_car_frames = lost_car_frames +1 if car is None else 0
            #允许小车在球不在时重锁
        

                
            #长时间丢失时进行解锁，等待重新锁定
            if lost_tennis_frames > LOST_MAX_FRAMES or lost_car_frames > LOST_MAX_FRAMES:
                print("UNLOCK目标长期丢失，解除锁定，等待重新锁定")
                target_tennis_id,target_car_id =None,None
                lost_tennis_frames = lost_car_frames = 0
            else:
                if (car is None) and (ball is not None) and car_detections:
                    bx,by = ball["center"]
                    nearest_car = min (car_detections,key= lambda c:(c["center"][0]-bx) **2 + (c["center"][1]-by)**2)
                    target_car_id = nearest_car["id"]
                    car = nearest_car
                    lost_car_frames=0
                    print(f"RELOCK小车丢失后冲所CARID ={target_car_id}TENNISID保持为{target_tennis_id}")



                # if(ball is not None) and (car is not None):
                #     dist_mm = pixel_distance_mm(car["center"],ball["center"],mm_per_pixel=mm_per_pixel)

                        
                #     if dist_mm > DIST_MM_THRESHOLD:
                #         send_control_command("up",speed=50)
                #         was_moving_forward = True  # 标记当前在前进
                #     else:
                #         send_control_command("stop",speed = 0)
                #         if was_moving_forward:
                #             send_control_command("grab",speed = 0)
                #             grabbing = True
                #             print("抓取完成")

                #         was_moving_forward = False
                #     # else:
                #     #     send_control_command("stop",speed= 0)
                #     #     send_control_command("grab",speed = 0)
                #     #     grabbing = True
                #     #     print("GRAB 已停止并且抓取")
                if (ball is not None) and (car is not None):
                    dist_mm = pixel_distance_mm(car["center"], ball["center"], mm_per_pixel=mm_per_pixel)

                    if dist_mm > DIST_MM_THRESHOLD:
                        # CAR_SPEED_MM_S = 150.0  # 小车速度 15 cm/s = 150 mm/s
                        # time_to_reach = dist_mm / CAR_SPEED_MM_S
                        # print(f"预计前进时间: {time_to_reach:.2f}s (距离 {dist_mm:.1f} mm)")
                        
                        # 发送一次前进命令，持续 time_to_reach 秒
                        if dist_mm > 650:  # 大于55cm
                            forward_time = 2.55
                        elif dist_mm > 600 and dist_mm <650:  
                            forward_time = 2
                        elif dist_mm > 550 and dist_mm <600:  
                            forward_time = 1.8
                        elif dist_mm > 500 and dist_mm <550:  
                            forward_time = 1.5
                        elif dist_mm > 400 and dist_mm <500:  
                            forward_time = 1.3
                        else:  
                            forward_time = 1
                        send_control_command("up", speed=50)
                        time.sleep(forward_time)
                        send_control_command("stop", speed=0)
                        # 前进完成后直接抓取（等同于“预约抓取”）
                        send_control_command("grab", speed=0)
                        grabbing = True
                        print("前进完成，执行抓取")
                    else:
                        # 如果已经很近，就直接抓取
                        send_control_command("stop", speed=0)
                        send_control_command("grab", speed=0)
                        grabbing = True
                        print("距离阈值内，直接抓取")


                    cv2.putText(frame,f"BALLID#{target_tennis_id} CARID#{target_car_id}",
                                (20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
                    cv2.putText(frame,f"Distance:{dist_mm:.1f}mm",
                                (20,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
                    print(f"distance(car#{target_car_id},ball#{target_tennis_id})= {dist_mm:.2f}mm")
                else:
                    #任一缺失，不移动
                    send_control_command("stop",speed=0)
        else:
            if mm_per_pixel is None:        
                cv2.putText(frame,"waiting for tennis",
                            (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,255),2)
                    

        if mm_per_pixel is not None:
            cv2.putText(frame,f"mm/px (fixed): {mm_per_pixel:.3f}",
                        (20,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1)& 0xFF
        if key ==ord('q'):
            break
        if key ==ord('r'):
            target_tennis_id = None
            target_car_id = None
            grabbing = False
            lost_tennis_frames = lost_car_frames = 0
            print("手动复位，等待再次锁定")
        # if key ==ord('c'):
        #     mm_per_pixel = None
        #     calibrated = False
        #     print("重新标定")
                
        if key == ord('t') :
            send_control_command("stop", speed=0)
            send_control_command("grab", speed=0)
            grabbing = True
            print("手动执行抓取")
        if key == ord('y') and not grabbing:
            send_control_command("stop", speed=0)
            send_control_command("release", speed=0)
            grabbing = True
            print("手动执行释放")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



            

