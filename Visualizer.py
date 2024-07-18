import cv2

class Visualizer:
    def __init__(self):
        self._field_seperator_color = (255, 255, 255)

    def draw_field_seperator(self, frame, x):
        top_left = [x - 10, 0]
        bottom_right = [x + 10, frame.shape[0]]
        # top_right = [x + 10, 0]
        # bottom_left = [x - 10, frame.shape[0]]

        # points = [top_left, top_right, bottom_right, bottom_left]
        cv2.rectangle(frame, top_left, bottom_right, self._field_seperator_color, 2)
        return frame

        # print(points)
        # opacity=0.2
        # overlay = frame.copy()
        # cv2.fillPoly(overlay, [points], (120, 255, 55))
        # cv2.polylines(overlay, [points], isClosed=True, color=(120, 255, 55), thickness=2)
        # cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        # return overlay