import cv2
import sys
from kcf import Tracker

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python run.py car.avi")
        exit(-1)
    cap = cv2.VideoCapture(sys.argv[1])
    tracker = Tracker()
    ok, frame = cap.read()
    if not ok:
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    #roi = (218, 302, 148, 108)
    #roi = (219, 298, 151, 111)
    print("roi = ", roi)
    tracker.init(frame, roi)
    while(cap.isOpened()):
        ok, frame = cap.read()
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        #cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 1)
            
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
