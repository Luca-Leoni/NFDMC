import torch

from torch import Tensor
from ..Archetypes import Flow, Diagrammatic, block_types

#------------------------------------------

class DiaChecker(Flow, Diagrammatic):
    """
    Transformation for chagning the vector in output from a general transformation back to a Diagram by doing specific transformation for every block type.
    """
    def __init__(self, last: bool = False):
        """
        Constructor

        Create the transformation layer that depending on its position in the flow can change the transformation that performs. In particular if it's the last operation it brings back the diagram in its original form after has been shuffled in the flow.

        IMPORTANT: this transformation is not invertible, so adding it the flow would not be able to compute the log probability of the model. Not a problem if using the reverse_kdl loss.

        Parameters
        ----------
        last
            Tells if its the last transformation or not.
        """
        super().__init__()
        
        self.__last = last

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Apply the wanted transformations to the batch

        Parameters
        ----------
        z
            Batch of vectors

        Returns
        -------
        tuple[Tensor, Tensor]
            Batch of diagrams and log det of the transformation, so zero
        """
        block_type = self.get_block_types()

        # If is last layer put the diagram in initial normal composition
        if self.__last:
            dia_comp   = self.get_dia_comp()

            x = torch.clone(z)
            z = x[:, dia_comp[0,0]:dia_comp[0,1]]
            for i in range(1, dia_comp.shape[0]):
                z = torch.cat( (z, x[:, dia_comp[i,0]:dia_comp[i,1]]), dim=1 )
                
            self.set_initial_comp()

        # perform wanted transformation to have a normal diagram
        for i, block in enumerate(self.get_dia_comp()):
            if block_type[i] == block_types.integer and self.__last:
                z[:, block[0]:block[1]] = torch.floor(z[:, block[0]:block[1]])
            elif block_type[i] == block_types.tm_ordered:
                z[:, block[0]+1:block[1]:2] += z[:, block[0]:block[1]:2]

        return torch.abs(z), torch.zeros(z.shape[0], device=z.device)
