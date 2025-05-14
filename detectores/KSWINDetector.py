from river.drift import KSWIN
from detectores.DetectorDriftBase import DetectorDriftBase

class KSWINDetector(DetectorDriftBase):
    def __init__(self, seed=None):
        super().__init__()
        self.detector = KSWIN(seed=seed)
        self.name = "_KSWIN"
