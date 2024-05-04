import cv2
import time

cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

width = int(cap.get(3))
height = int(cap.get(4))
fcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test.avi', fcc, 60.0, (width, height))
recording = False

while(1):
    ret, frame = cap.read()
    hms = time.strftime('%H:%M:%S', time.localtime())

    cv2.putText(frame, str(hms), (0, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255))

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == ord('r'):
        path = 'test_' + str(hms) + '.avi'
        writer = cv2.VideoWriter(path, fcc, 60.0, (width, height))

    if recording:
        writer.write(frame)

    if k == ord('e'):
        print('record end')
        writer.release()

cap.release()
cv2.destroyAllWindows()