import torch

from torch import Tensor
from ..Archetypes import Flow, Diagrammatic

#-----------------------------------

class PermuteRandom(Flow):
    """
    Flows that randomly permutes the elements inside the input variables
    """
    def __init__(self, var_dim: int):
        """
        Constructor

        Parameters
        ----------
        var_dim
            Dimensions of the input variable
        """
        super().__init__()

        self.register_buffer("_per_for", torch.randperm(var_dim))
        self.register_buffer("_per_inv", torch.argsort(self._per_for)) #pyright: ignore


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input data

        Returns
        -------
        tuple[Tensor, Tensor]
            Output data and log det of the transformation, in this case is 0
        """
        return z[:, self._per_for], torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse of the transformation

        Parameters
        ----------
        z
            Transformed input variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformerd variable and log det of the inverse, 0 also in this case
        """
        return z[:, self._per_inv], torch.zeros(z.shape[0], device=z.device)


class Swap(Flow):
    """
    Simple permuation flow that swaps the entries inside the random variable
    """
    def __init__(self):
        """
        Constructor

        A permutation flow that swaps the entries inside the random variable
        """
        super().__init__()

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Takes the random variable z and flips it over swapping the entries

        Parameters
        ----------
        z
            Batch of variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Flipped variables and log det of the transformation, so zero
        """
        return torch.flip(z, dims=(1,)), torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse of the swapping, which is the swapping itself

        Parameters
        ----------
        z
            Batch of variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Flipped variables and log det of the transformation, so zero
        """
        return torch.flip(z, dims=(1,)), torch.zeros(z.shape[0], device=z.device)


class PermuteTimeBlock(Flow, Diagrammatic):
    """
    Diagrammatic Flow that permutes the Couples inside an order time block in a random way.
    """
    def __init__(self, block_name: str):
        """
        Constructor

        Construct a permutation diagrammatic flow to permute the couples inside a time ordered block of the diagram in a random way. In particular an order time block is assumed to look as:
            [creation_time_1, destruction_time_1, crea_tm_2, dest_time_2, ...]
        so that the couples needs to stay toghether and the number of element inside the block are also even.

        Parameters
        ----------
        block_name
            Name of the time ordered block to work on
        """
        super().__init__()

        block = self.get_block(block_name)

        self.__n_couple = int(self.get_dia_comp()[block][1] - self.get_dia_comp()[block][0]) // 2

        self.__b = block
        self.__permutation = torch.randperm(self.__n_couple)
        self.__inverse     = torch.argsort(self.__permutation)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Takes as inputs the batch of diagrams and return the version with the couples in the wanted block shuffled.

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Shuffled diagrams and log det of the permutation, so zero
        """
        comp = self.get_dia_comp()
        start = comp[self.__b, 0]
        end   = comp[self.__b, 1]

        times = z[:, start:end].reshape(z.shape[0], self.__n_couple, 2)

        z[:, start:end] = times[:, self.__permutation, :].flatten(start_dim=1)

        return z, torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """    
        Apply the inverse permutation to the couples inside the designed block

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
             Shuffled diagrams and log det of the permutation, so zero
        """
        comp = self.get_dia_comp()
        start = comp[self.__b, 0]
        end   = comp[self.__b, 1]

        times = z[:, start:end].reshape(z.shape[0], self.__n_couple, 2)

        z[:, start:end] = times[:, self.__inverse, :].flatten(start_dim=1)

        return z, torch.zeros(z.shape[0], device=z.device)

    def get_permutation(self) -> Tensor:
        return self.__permutation


class SwapDiaBlock(Flow, Diagrammatic):
    """
    Diagrammatic Flow that swaps the position of two blocks inside the diagram composition.
    """
    def __init__(self, block1_name: str, block2_name: str):
        """
        Constructor

        Construct a diagrammatic permuation flow that swaps the position of two wanted blocks inside the diagram

        Parameters
        ----------
        block1_name
            Name of the first block
        block2_name
            Name of the second block
        """
        super().__init__()

        self.__bn = [block1_name, block2_name]
        self.__b = torch.tensor([self.get_block(block1_name), 
                                 self.get_block(block2_name)])


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Takes the batch of diagrams and swaps the wanted blocks of the diagrams

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Shuffled diagrams and log det of the permutation, so zero
        """
        split_ord = self.get_dia_comp()[self.__b].flatten().msort()

        zb = z[:, :split_ord[0]]
        z1 = z[:, split_ord[0]:split_ord[1]]
        zm = z[:, split_ord[1]:split_ord[2]]
        z2 = z[:, split_ord[2]:split_ord[3]]
        ze = z[:, split_ord[3]:]

        self.swap_blocks(self.__bn[0], self.__bn[1])
        return torch.cat((zb, z2, zm, z1, ze) , dim=1), torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse of the swapping, which is the same transformation   

        Parameters
        ----------
        z
            batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Shuffled diagrams and log det of the permutation, so zero
        """
        return self.forward(z)
