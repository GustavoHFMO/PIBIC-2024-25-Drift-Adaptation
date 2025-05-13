from river.drift.binary import HDDM_A
from detectores.DetectorDriftBase import DetectorDriftBase


class HDDM_ADetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = HDDM_A()
        self.name = "_HDDMa"

