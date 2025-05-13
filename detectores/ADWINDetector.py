from river.drift import ADWIN
from detectores.DetectorDriftBase import DetectorDriftBase

class ADWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = ADWIN()
        self.name = "_ADWIN"
