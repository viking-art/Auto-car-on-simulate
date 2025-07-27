import socket
import time
import cv2
import numpy as np
import json
import base64
import torch
from Test.unet import UNet
from Test.modelSegmentation import *

import torch
import math
from model import build_unet

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
# s.connect(('127.0.0.1', port))

def PID(err, Kp, Ki, Kd):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()

    P = Kp * err
    D = Kd * (err - err_arr[1]) / (delta_t + 1e-7)
    I = Ki * np.sum(err_arr) * delta_t
    angle = P + I + D

    if abs(angle) > 25:
        angle = np.sign(angle) * 25
        print(angle)

    return int(angle)
def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove

#segmentation
def segment(img):
    img = cv2.cvtColor(imgage, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 160))
    x = torch.from_numpy(img)
    x = x.to(device)
    x = x.transpose(1, 2).transpose(0, 1)
    x = x / 255.0
    x = x.unsqueeze(0).float()
    pred_y = net(x)
    pred_y = torch.sigmoid(pred_y)
    pred_y = pred_y[0]
    pred_y = pred_y.squeeze()
    pred_y = pred_y > 0.5
    pred_y = pred_y.cpu().numpy()
    pred_y = np.array(pred_y, dtype=np.uint8)
    pred_y = pred_y * 255
    pred_pr = pred_y[:,:]
    return pred_pr,img

#noise out lane
def morphology(b_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=5)
    b_img = cv2.dilate(b_img, kernel,iterations=3)
    b_img = remove_small_contours(b_img)
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=5)
    return b_img


logger = open("log.txt", 'w')
if __name__ == "__main__":
    net = UNet(3, 3)
    net = build_unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load("output_1.pth", map_location=device))
    half = device.type != 'cpu'
    prev_frame_time = 0
    new_frame_time = 0
    angle = 5
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe

            speed = 30
            if angle>=-2 and angle<=2:
                speed=80
            if angle==0:
                speed=120
            if angle<-2 or angle>2:
                speed= 15
    

            # print(angle)
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)



            # Receive data from server
            data = s.recv(100000)
            # print(data)
            try:
                data_recv = json.loads(data)
            except:
                logger.writelines('system crash')
                continue

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            print("current_angle: ", current_angle)
            print("current_speed: ", current_speed)
        
            # Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            imgage = cv2.resize(imgage, (320, 160))

            with torch.no_grad():
                b_img, resizeed_img = segment(imgage)
                b_img = morphology(b_img)

                arr=[]
                k=110
                lineRow = b_img[k,:]
                for x,y in enumerate(lineRow):
                    if y == 255:
                      arr.append(x)

                arrmax = max(arr,default=0)
                arrmin = min(arr,default=0)
                center= int((arrmax+arrmin)/2)
                angle1= math.degrees(math.atan((center-b_img.shape[1]/2)/(b_img.shape[0]-80)))
                #print("Error: ",angle1)
                cv2.circle(b_img, (arrmin, k), 5, (0, 255, 255), 5)
                cv2.circle(b_img, (arrmax, k), 5, (0, 255, 255), 5)
                cv2.line(b_img, (center, k), (int(b_img.shape[1] / 2), b_img.shape[0]), (0, 255, 255), 5)

                pre_t = time.time()
                # print(pre_t)
                err_arr = np.zeros(5)
                error = 160 - center
                print("Error:",error)
                angle = -PID(error, 0.32, 0.19, 0.0000000000001)

                print("sendback_angle: ", angle)
                print("sendback_speed: ", speed)
                print("---------------------------------------")

                #cv2.circle(imgage, (center, 120), 5, (0, 0, 255), -1)



                cv2.imshow("binary", b_img)
                cv2.imshow("IMG", imgage)

                #print("lineRow", lineRow)
            key = cv2.waitKey(1)

    finally:
        print('closing socket')
        s.close()

