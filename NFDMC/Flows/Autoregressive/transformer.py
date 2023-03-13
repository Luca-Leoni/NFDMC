from ..flow import Flow

class Affine(Flow):
    def __init__(self, conditioner) -> None:
        super().__init__()
