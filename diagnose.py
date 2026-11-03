import cv2

for p in [r"data\raw\Animals_1of2\Animals\1. Dog\MVI_8560.MP4",
          r"data\raw\Animals_1of2\Animals\1. Dog\MVI_3002.MOV"]:
    c = cv2.VideoCapture(p)
    print(p.split("\\")[-1], int(c.get(3)), "x", int(c.get(4)), "| fps", c.get(5),
          "| frames", int(c.get(7)))