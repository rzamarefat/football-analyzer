import torch

class Configuration:
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEGMENTOR_CKPT_PATH = r"C:\Users\ASUS\Desktop\github_projects\Football\runs\segment\train\weights\best.pt"
    TRACKER_CKPT_PATH = r"C:\Users\ASUS\Desktop\github_projects\Football\yolov8x.pt"

    COUNTER_FOR_WARM_UP_STAGE = 1