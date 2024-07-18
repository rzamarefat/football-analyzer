import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from collections import Counter
from matplotlib import pyplot as plt
from Configuration import Configuration as CONFIG

class WarmUpHandler:
    def __init__(self, tracker):
        self._tracker = tracker
        self._device = CONFIG.DEVICE

    def _extract_dominant_color(self, image, k=2):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[0:int(image.shape[0]/2),:]
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=1)
        kmeans.fit(pixels)
        counts = Counter(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
        labels = kmeans.labels_
        # top_half_image = image[0:int(image.shape[0]/2),:]

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(image.shape[0],image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        dominant_color = kmeans.cluster_centers_[player_cluster]

        return dominant_color

    def _cluster_images_by_color(self, images, n_clusters=3):
        dominant_colors = []
        for img in images:
            dominant_color = self._extract_dominant_color(img)
            dominant_colors.append(dominant_color)
        
        dominant_colors = np.array(dominant_colors)
        
        self._kmeans = KMeans(n_clusters=n_clusters)
        labels = self._kmeans.fit_predict(dominant_colors)
        
        return labels

    def _get_people_clusters(self, frame):
        res = self._tracker.track(frame, device=self._device, verbose=False)
        boxes = res[0].boxes.xyxy
        boxes = boxes.to("cpu").tolist()
        classes = [int(i) for i in res[0].boxes.cls.to("cpu").tolist()]
        person_crops = []
        for box, cls_ in zip(boxes, classes):
            if cls_ == 0:
                tlx = int(box[0])
                tly = int(box[1])
                brx = int(box[2])
                bry = int(box[3])

                person_crops.append(frame[tly:bry,tlx:brx,:])
        
        self._cluster_images_by_color(person_crops)
    
    def get_person_cluster(self, person_crop):
        dominant_color = self._extract_dominant_color(person_crop)
        return self._kmeans.predict([dominant_color]).item()
    
    def __call__(self, frame):
        self._get_people_clusters(frame)

    



