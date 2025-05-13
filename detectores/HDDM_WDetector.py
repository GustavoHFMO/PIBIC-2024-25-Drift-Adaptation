from river.drift.binary import HDDM_W
from detectores.DetectorDriftBase import DetectorDriftBase


class HDDM_WDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = HDDM_W()
        self.name = "_HDDMw"

