import cv2
import argparse

parser = argparse.ArgumentParser(description="Check index of eye-in-hand camera (OpenCV)")

# Adding arguments
parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index.")
args = parser.parse_args()

cam_index = args.camera_index
cam = cv2.VideoCapture(cam_index)

print("Camera index: %d" % cam_index)

while True:
    _, frame = cam.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()