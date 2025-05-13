from river.drift.binary import EDDM
from detectores.DetectorDriftBase import DetectorDriftBase

class EDDMDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = EDDM()
        self.name = "_EDDM"
