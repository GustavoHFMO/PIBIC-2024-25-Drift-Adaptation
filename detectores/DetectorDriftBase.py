from abc import ABC

class DetectorDriftBase(ABC):
    def __init__(self):
        pass

    def atualizar(self, erro):
        self.detector.update(erro)

    def drift_detectado(self):
        return self.detector.drift_detected