import cv2

class Visualizer:
    def __init__(self):
        self._field_seperator_color = (255, 255, 255)
        
        self._people_cluster_color = {
            "refree": (0, 0, 0),
            "team1": (0, 0, 255),
            "team2": (255, 255, 255),
        }

    def draw_field_seperator(self, frame, x):
        top_left = [x - 10, 0]
        bottom_right = [x + 10, frame.shape[0]]
        cv2.rectangle(frame, top_left, bottom_right, self._field_seperator_color, 2)
        return frame


    def draw_cluster_on_people(self, frame, people_cluster_info, boxes, ids, classes):
        for box, id_, cls_ in zip(boxes, ids, classes):
            if cls_ == 0:

                if people_cluster_info[id_] == 0:
                    color = self._people_cluster_color["refree"]
                elif people_cluster_info[id_] == 1:
                    color = self._people_cluster_color["team1"]
                else:
                    color = self._people_cluster_color["team2"]
                    
                tlx = int(box.to("cpu").tolist()[0])
                tly = int(box.to("cpu").tolist()[1])
                brx = int(box.to("cpu").tolist()[2])
                bry = int(box.to("cpu").tolist()[3])
                cv2.rectangle(frame, (tlx, tly), (brx, bry), color, 2)
        
        return frame



    