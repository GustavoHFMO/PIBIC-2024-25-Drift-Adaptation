from river.drift.binary import DDM
from detectores.DetectorDriftBase import DetectorDriftBase

class DDMDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = DDM()
        self.name = "_DDM"

    def atualizar(self, erro):
        erro_ajustado = 0 if erro < 0 else 1 if erro > 1 else erro
        self.detector.update(erro_ajustado)