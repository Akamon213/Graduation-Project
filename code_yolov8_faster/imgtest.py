#coding:utf-8
from ultralytics import YOLO
import cv2

# 所需加载的模型目录
# path = 'models/best2.pt'

path = r'runs\train\exp4\weights\best.pt'
# path = r'D:\Project\PCB_v2\PCB\yolov8_v2\code_yolov8_faster\legacy_weights\YOLOv8s\weights\best.pt'

# 需要检测的图片地址
img_path = r"D:\Project\PCB\data\data\PCB_DATASET\VOC\yolo\train\images\train\04_missing_hole_03.jpg"
# img_path = r"D:\Project\PCB_v2\PCB\VOC_PCB\yolo\val\images\val\l_light_04_mouse_bite_05_3_600.jpg"

# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)


# 检测图片
results = model(img_path)
res = results[0].plot()
# res = cv2.resize(res,dsize=None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)