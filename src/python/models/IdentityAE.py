## Author: Benjamin Bissendorf

from python.models.AEModel import AEModel


class IdentityAE(AEModel):
    def load(self) -> None:
        pass
    def getName(self) -> str:
        return "IdentityAE"
    def encode(self, img) -> list[float]:
        return img
    def decode(self, latent: list[float]) -> []:
        return latent