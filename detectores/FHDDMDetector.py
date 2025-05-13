from river.drift.binary import FHDDM
from detectores.DetectorDriftBase import DetectorDriftBase

class FHDDMDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = FHDDM()
        self.name = "_FHDDM"
