import numpy as np
import cv2 as cv
import Person
import time

try:
    log = open('data.txt', "w")
except:
    print("No open file!")

# Biến đếm ra vào vào
cnt_up = 0
cnt_down = 0

# Import Camera
# cap = cv.VideoCapture(0)

# import bằng video để test
cap = cv.VideoCapture('test.avi')

# Config thuộc tính màn hình
# cap.set(3,1280) #Width
# cap.set(4,750) #Height

# In các thuộc tính capture của bản điều khiển
for i in range(19):
    print(i, cap.get(i))

h = 480 # chiều cao
w = 640  # chiều rộng
frameArea = h * w  # Tính diện tích cà frame ảnh

# vùng ngưỡng
# areaTH = frameArea / 250
areaTH = frameArea / 250
print('Area Threshold', areaTH)

# Dòng vào ra
line_up = int(2 * (h / 5))  # 192
line_down = int(3 * (h / 5))  # 288

up_limit = int(1 * (h / 5))
down_limit = int(4 * (h / 5))

print("Red line y:", str(line_down))
print("Blue line y:", str(line_up))
# print("Yello line y:", str(down_limit))
# print("Green line y:", str(up_limit))

line_down_color = (255, 0, 0)  # Config màu đỏ cho dòng ra
# down_limit = (255, 255, 0)       #Màu vàng


line_up_color = (0, 0, 255)  # Config màu xanh cho dòng vào
# up_limit = (0, 255, 0)       #Màu xanh lá

# Chia vùng
pt1 = [0, line_down]
pt2 = [w, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)  # Tạo mảng
pts_L1 = pts_L1.reshape((-1, 1, 2))  # Định hình lại mảng thành mảng 1 chiều

pt3 = [0, line_up]
pt4 = [w, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [w, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))

pt7 = [0, down_limit]
pt8 = [w, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Thuật toán phân đoạn nền, phát hiện bóng
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)

# Thành phần của bộ lọc bóng
kernelOp = np.ones((3, 3), np.uint8)  # Hàm ones sẽ trả về 1 cho mảng có kích thước 3x3
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while (cap.isOpened()):
    ##for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    ret, frame = cap.read()
    ##    frame = image.array

    for i in persons:
        i.age_one()  # age every person one frame
    #########################
    #   Tiền sử lý  #
    #########################

    # Áp dụng phép trừ nền
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    # Dùng phép nhị nhân để loại bỏ cái bóng (chuyển sang ảnh xám)
    try:
        ret, imBin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)  # Dùng ngưỡng màu để loại bỏ bóng
        ret, imBin2 = cv.threshold(fgmask2, 200, 255, cv.THRESH_BINARY)

        # Khởi động -> xóa mòn -> giãn nở ảnh để loại bỏ những phần mờ
        # Hàm morphologyEx là phép biến đổi hình thái nâng cao bằng cách sử dụng sự xói mòn và giãn nở
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)

        # Đóng -> xóa mòn -> giãn nở ảnh -> để nối những vùng trắng lại
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelCl)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
    except:
        # print('EOF')
        print('UP:', cnt_up)
        print('DOWN:', cnt_down)
        break
    #################
    #   Đường viền   #
    #################

    # RETR_EXTERNAL chỉ trả về các cờ bên ngoài cùng cực. Tất cả các đường nét con đều bị loại bỏ.
    # Dùng hàm findContours để tìm các đường bao từ một hình ảnh nhị phân
    # Đường viền là thứ để chúng ta phân tích hình dạng, phát hiện và nhận dạng đối tượng.
    contours0, hierarchy = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)  # Tính diện tích đường bao
        if area > areaTH:
            #################
            #   TRACKING    #
            #################

            # Cho phép nhiều người ra vào cùng một lúc

            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv.boundingRect(cnt) #Trả về hình chữ nhật

            new = True
            if cy in range(up_limit, down_limit):
                for i in persons:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        # đối tượng ở gần với một đối tượng đã được phát hiện trước đó
                        new = False
                        i.updateCoords(cx, cy)  # cập nhật tọa độ trong đối tượng và đặt lại tuổi
                        if i.going_UP(line_down, line_up) == True:
                            cnt_up += 1;
                            print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            log.write("ID: " + str(i.getId()) + ' crossed going up at ' + time.strftime("%c") + '\n')
                        elif i.going_DOWN(line_down, line_up) == True:
                            cnt_down += 1;
                            print("ID:", i.getId(), 'crossed going down at', time.strftime("%c"))
                            log.write("ID: " + str(i.getId()) + ' crossed going down at ' + time.strftime("%c") + '\n')
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # Xóa khỏi danh sách person
                        index = persons.index(i)
                        persons.pop(index)
                        del i  # Giải phóng vùng nhớ
                if new == True:
                    p = Person.MyPerson(pid, cx, cy, max_p_age)
                    persons.append(p)
                    pid += 1

            # Vẽ khung nhận diện bằng phép tịnh tiến
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # END for cnt in contours0

    # Vẽ
    for i in persons:
        ##        if len(i.getTracks()) >= 2:
        ##            pts = np.array(i.getTracks(), np.int32)
        ##            pts = pts.reshape((-1,1,2))
        ##            frame = cv.polylines(frame,[pts],False,i.getRGB())
        ##        if i.getId() == 9:
        ##            print str(i.getX()), ',', str(i.getY())
        cv.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv.LINE_AA)

    # Hiển thị
    str_up = 'Up: ' + str(cnt_up)
    str_down = 'Down: ' + str(cnt_down)

    # Dùng phép chống xói mòn anh để loại bỏ vùng biên ko xác định
    frame = cv.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
    cv.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    cv.imshow('Counter', frame)
    #cv.imshow('Mask', mask)

    # Nhấn Esc kết thoát chương trình
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Giải phóng vùng nhớ

log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
