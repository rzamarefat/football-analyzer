from typing import Any
from ultralytics import YOLO
from Configuration import Configuration as CONFIG
import cv2
import numpy as np
from WarmUpHandler import WarmUpHandler
from collections import Counter


class Analyzer:
    def __init__(self, do_visualize=True):
        self._do_visualize = do_visualize
        if self._do_visualize:
            from Visualizer import Visualizer
            self._visualizer = Visualizer() 

        self._field_segmentor = YOLO(CONFIG.SEGMENTOR_CKPT_PATH)
        self._tracker = YOLO(CONFIG.TRACKER_CKPT_PATH)
        self._device = CONFIG.DEVICE

        self._warm_up_stage = True
        self._counter_for_warm_up_stage = 0
        self._warm_up_handler = WarmUpHandler(self._tracker)

        self._people_cluster_info = {}
        self._people_cluster_history = {}

        self._field_seperator = None

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
        res = self._field_segmentor.predict(frame, device=self._device, verbose=False)

        for mask, cls_ in zip(res[0].masks.data, res[0].boxes.cls):
            mask = mask.to("cpu").numpy()
            if int(cls_.item()) == 0:
                self._field_seperator = self._find_rightmost_white_pixel(frame, mask)
                
            elif int(cls_.item()) == 1:
                continue
                

            

    def _handle_visualization(self, frame):
        if self._field_seperator is not None:
            frame = self._visualizer.draw_field_seperator(self._org_frame, self._field_seperator[0])
        frame = self._visualizer.draw_cluster_on_people(frame, self._people_cluster_info, self._track_boxes, self._track_ids, self._track_classes)

        return frame

        


        
            
    def _get_players(self):
        pass

    def _get_people(self):
        for box, class_, id_ in zip(self._track_boxes, self._track_classes, self._track_ids):
            if class_ == 0:
                tlx = int(box.to("cpu").tolist()[0])
                tly = int(box.to("cpu").tolist()[1])
                brx = int(box.to("cpu").tolist()[2])
                bry = int(box.to("cpu").tolist()[3])
                person_crop = self._org_frame[tly:bry,tlx:brx,:]
                person_cluster = self._warm_up_handler.get_person_cluster(person_crop)

                if id_ not in self._people_cluster_history.keys():
                    self._people_cluster_history[id_] = [person_cluster]
                else:
                    self._people_cluster_history[id_].append(person_cluster)

                self._people_cluster_info[id_] = person_cluster

                # counter = Counter(self._people_cluster_history[id_])
                # most_common_cluster_of_person, _ = counter.most_common(1)[0]
                # self._people_cluster_info[id_] = most_common_cluster_of_person

    def _get_refrees(self):
        pass

    def _track(self, frame):
        self._frame = frame.copy()
        res = self._tracker.track(frame, device=self._device, verbose=False)
        
        boxes = res[0].boxes.xyxy
        classes = [int(i) for i in res[0].boxes.cls.to("cpu").tolist()]
        ids = [int(i) for i in res[0].boxes.id.to("cpu").tolist()]

        return boxes, classes, ids

    def _do_warm_up_stage(self):
        self._warm_up_handler(self._org_frame)
        
                
        
    def __call__(self, frame):
        self._org_frame = frame.copy()
        if self._warm_up_stage and self._counter_for_warm_up_stage <= CONFIG.COUNTER_FOR_WARM_UP_STAGE:
            self._do_warm_up_stage()
            self._counter_for_warm_up_stage += 1
    
        self._field_composition = self._get_field_composition(frame)
        self._track_boxes, self._track_classes, self._track_ids = self._track(frame)

        self._get_people()


        print(self._people_cluster_history)
        print(self._people_cluster_info)


        if self._do_visualize:
            print(self._do_visualize)
            visualized_frame = self._handle_visualization(self._org_frame)
            cv2.imwrite("visualized_frame.jpg", visualized_frame)
            print("dasdasdasdasd")



        


        


        


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