from typing import Any
from ultralytics import YOLO
from Configuration import Configuration as CONFIG
import cv2
import numpy as np



class Analyzer:
    def __init__(self, do_visualize=True):
        self._do_visualize = do_visualize
        if self._do_visualize:
            from Visualizer import Visualizer
            self._visualizer = Visualizer() 

        self._field_segmentor = YOLO(CONFIG.SEGMENTOR_CKPT_PATH)
        self._tracker = YOLO(CONFIG.TRACKER_CKPT_PATH)
        self._device = CONFIG.DEVICE

    @staticmethod
    def _find_rightmost_white_pixel(image, frame):
        resized_tensor = cv2.resize(frame, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        white_pixels = np.argwhere(resized_tensor == 1)        
        if white_pixels.size == 0:
            return None  

        x_coords = white_pixels[:, 1]
        rightmost_x = np.max(x_coords)
        rightmost_y = white_pixels[white_pixels[:, 1] == rightmost_x][:, 0]
        rightmost_pixel = (rightmost_x, rightmost_y[0])
        
        return rightmost_pixel

    def _get_field_composition(self, frame):
        res = self._field_segmentor.predict(frame, device=self._device)

        for mask, cls_ in zip(res[0].masks.data, res[0].boxes.cls):
            mask = mask.to("cpu").numpy()
            if int(cls_.item()) == 0:
                field_seperator = self._find_rightmost_white_pixel(frame, mask)
                
            elif int(cls_.item()) == 1:
                continue

            
            frame = self._visualizer.draw_field_seperator(frame, field_seperator[0])

            cv2.imwrite("frame.jpg", frame)
            
    def _get_players(self):
        pass

    def _get_people(self):
        pass

    def _get_refrees(self):
        pass

    def _track(self, frame):
        return self._tracker.track(frame, device=self._device)



    def __call__(self, frame):
        field_composition = self._get_field_composition(frame)
        res = self._track(frame)
        self._get_people()

        


        


        


if __name__ == "__main__":
    analyser = Analyzer()


    cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\github_projects\Football\videos\vid_1.avi")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
        
            res = analyser(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else: 
            break
        
    cap.release()
    cv2.destroyAllWindows()