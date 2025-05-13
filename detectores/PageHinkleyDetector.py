from river.drift import PageHinkley
from detectores.DetectorDriftBase import DetectorDriftBase

class PageHinkleyDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = PageHinkley()
        self.name = "_PageHinkley"
