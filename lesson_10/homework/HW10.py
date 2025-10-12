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

"""
After trying to track cars, I realized it was hard to reliably trigger conditions that reveal each tracker's strengths and weaknesses. So I switched to tracking my handheld device (orange Rabbit R1) to stress specific failure modes under controlled challenges.

KCF is fast but less accurate; CSRT is slower but more accurate.
KCF tends to lose the target more easily; CSRT is usually more robust. However, in my experiment KCF recovered from occlusions better, while CSRT drifted over time (it started tracking my finger instead of the Rabbit R1).
Note: although CSRT is commonly considered better at handling occlusions and recovery, this result is just an observation for my case, not a general claim.
"""