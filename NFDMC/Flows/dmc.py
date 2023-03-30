import torch
import torch.nn as nn

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


class OrderTime(Flow, Diagrammatic):
    """
    Flow that can be inserted inside the flow in order to ensure that the time ordered blocks remains effectivelly ordered. In particular assumes that the flow was made so that the times inside the diagram are positive and so that we can simply order them by summing to the destruction time the creation one.
    """
    def __init__(self, trainable: bool = False):
        """
        Constructor

        Creates a diagrammatic flow that allows for the time ordered block to remain ordered inside the structure by doing the following operation
            .. math::
                z_i^d' = z_i^d + z_i^c
        So that also the log determinant of the transformation is simply zero.

        Raises
        ------
        RuntimeWarning:
            If no time ordered blocks are present then there is no point in using this flow.
        """
        super().__init__()

        # Gather all the time ordered blocks
        self.__blocks = []
        for i, type in enumerate(self.get_block_types()):
            if type == block_types.tm_ordered:
                self.__blocks.append(i)

        if len(self.__blocks) == 0:
            raise RuntimeWarning("No time ordered blocks are present there is no point in inserting a OrderTime inside the flow!")

        if trainable:
            self.delta = nn.Parameter(torch.ones(len(self.__blocks), 1) * 10)
        else:
            self.register_buffer("delta", torch.ones(len(self.__blocks), 1) * 10)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Order the creationa nd annhilation time inside the diagram by simply adding one to the other.

        Parameters
        ----------
        z
            Batch with the diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Orderd diagrams with log det of the transformation, so zero
        """
        tm_fly  = self.get_block_from("tm_fly", z)
        log_det = torch.zeros(z.shape[0], device=z.device)
        x       = torch.clone(z)

        for j, i in enumerate(self.__blocks):
            beg = self.get_dia_comp()[i,0]
            end = self.get_dia_comp()[i,1]

            scaled_tm_c = z[:, beg:end:2] / self.delta[j]
            scaled_tm_d = z[:, beg+1:end:2] / self.delta[j]

            x[:, beg:end:2] = tm_fly/(1 + torch.exp(-z[:, beg:end:2] / self.delta[j]
))
            x[:, beg+1:end:2] = (tm_fly - x[:, beg:end:2])/(1 + torch.exp(-z[:, beg+1:end:2] / self.delta[j])) + x[:, beg:end:2]

            # Setup consistend log det computation
            log_det += torch.sum(torch.log(tm_fly/self.delta[j]) - torch.logsumexp(self.__logsumexp_setup(scaled_tm_c), dim=2), dim=1)
            log_det += torch.sum(torch.log((tm_fly - x[:, beg:end:2])/self.delta[j]) - torch.logsumexp(self.__logsumexp_setup(scaled_tm_d), dim=2), dim=1)

        bad = torch.isinf(log_det)
        if bad.any():
            print(f"The bad diagrams that makes things crush are:\n {x[bad, :]}")

        return x, log_det

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inerse of the transformation, so that basically instead of summing the creation times we subtract them to the destruction ones.

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Unordered diagrams and log det of the transformation, so zero
        """
        tm_fly  = self.get_block_from("tm_fly", z)
        log_det = torch.zeros(z.shape[0], device=z.device)

        for j, i in enumerate(self.__blocks):
            beg = self.get_dia_comp()[i,0]
            end = self.get_dia_comp()[i,1]

            tm_c = self.get_block_from(i, z, step=2).clone()
            tm_d = self.get_block_from(i, z, bias_l=1, step=2).clone()

            z[:, beg:end:2] = -self.delta[j] * torch.log(tm_fly/tm_c - 1)
            z[:, beg+1:end:2] = -self.delta[j] * torch.log((tm_fly - tm_c)/(tm_d - tm_c) - 1)

            # Setup consistend log det computation
            log_det += torch.sum(torch.log(tm_fly/self.delta[j]) - torch.logsumexp(self.__logsumexp_setup(z[:, beg:end:2]/self.delta[j]), dim=2), dim=1)
            log_det += torch.sum((tm_fly - z[:, beg:end:2])/self.delta[j] - torch.logsumexp(self.__logsumexp_setup(z[:, beg+1:end:2]/self.delta[j]), dim=2), dim=1)


        return z, -log_det


    def __logsumexp_setup(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], z.shape[1], 1).repeat(1,1,3)
        z[:,:,1] = 0.6931471805599453
        z[:,:,2] *= -1

        return z
