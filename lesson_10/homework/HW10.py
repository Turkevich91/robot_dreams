import cv2

# cap = cv2.VideoCapture(r"C:\Users\Public\Videos\4K Road traffic video for object detection and tracking - free d.mp4")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
bbox = cv2.selectROI("init", frame, False, False)  # select object with mouse
cv2.destroyWindow("init")

tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

for i in range(750):
    ok, frame = cap.read()
    if not ok: break
    ok, bbox = tracker.update(frame)
    if ok:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

