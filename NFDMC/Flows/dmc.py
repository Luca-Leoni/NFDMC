import torch
import torch.nn as nn

from torch import Tensor
from ..Archetypes import Flow, Diagrammatic, block_types, Transformer, LTransformer
from ..Modules.nets import RealMVP
from .permutation import SwapDiaBlock

#------------------------------------------

class BCoupling(Flow, Diagrammatic):
    """
    Coupling flow that allows for the application of a certain transformation on a selected block of the diagram.
    """
    def __init__(self, block: int | str, trans: Transformer, hidden_size: int = 100):
        """
        Construct

        Diagrammatic flow that allows for the application of a specific transformation on a selected block of the diagram using a normal Coupling architecture.
        The idea is to swap the selected block to the bottom of the diagram and then apply a split so to apply the transformation on the last part of the block influenced by all the other elements.
        In particular the block automatically defines the conditioner as a RealMVP NN with input and output sizes defined by the diagram compoisition and the transformer parameters needed.

        Parameters
        ----------
        block
            Index or name of the block on which apply the transformation
        trans
            Transformer to use on the block
        hidden_size
            Size of the hidden layer of the RealMVP used as conditioner
        """
        super().__init__()

        # Take the right index
        if isinstance(block, str):
            block = self.get_block(block)

        # Define the split list based on the diagram sizes
        dia_size = int(torch.sum(self.get_blocks_lenght()))
        blo_size = int(self.get_blocks_lenght()[block])
        even     = blo_size % 2

        self.__split = [dia_size - (blo_size // 2) - even, blo_size // 2 + even]

        # Defines the transformation to swap the block in the right place
        self.__swap = SwapDiaBlock(block, -1) 

        # Define the conditioner based on the split
        self.__cond = RealMVP(self.__split[0], hidden_size, hidden_size, self.__split[1] * trans.trans_features)

        # Save transformer
        self.__trans = trans


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module

        Apply the transformation to the batch of diagrams and returns the log det

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h)), dim=1))
        return  z, self.__trans.log_det(z2, h)


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the inverse of the transformation to the batch and returns the log det of the inverse

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        z2 = self.__trans.inverse(z2,h)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h)


class OBCoupling(Flow, Diagrammatic):
    """
    Coupling Diagrammatic Flow that act on the order block
    """
    def __init__(self, trans: Transformer, hidden_size: int = 100) -> None:
        """
        Constructor

        Generates a Diagrammatic Coupling flow that acts on the order of the diagram specifically. It automatically computes the upper limit for the order

        Parameters
        ----------
        trans
            Transformer to use
        hidden_size
            Size of the hidden layer for the conditioner
        """
        super().__init__()

        dia_size = int(torch.sum(self.get_blocks_lenght()))

        self.__split = [dia_size - 1, 1]

        # Defines the transformation to swap the block in the right place
        self.__swap  = SwapDiaBlock("order", -1) 

        # Define the conditioner based on the split
        self.__cond  = RealMVP(self.__split[0], hidden_size, hidden_size, trans.trans_features)

        # Save transformer
        self.__trans = trans

    
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Overrride of the torch.nn.Module method

        Transform the order of the diagram and computes the log determinant of the transformation

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h  = self.__cond(z1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h)), dim=1))
        return  z, self.__trans.log_det(z2, h)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply inverse transformation to the order of the diagram

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log determinant of the inverse
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h  = self.__cond(z1)
        z2 = self.__trans.inverse(z2, h)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h)


class TFBCoupling(Flow, Diagrammatic):
    """
    Diagrammatic Coupling Flow that acts specifically on the tm_fly block
    """
    def __init__(self, trans: Transformer, hidden_size: int = 100) -> None:
        """
        Constructor

        Craetes a Diagrammatic Flow that apply the selected transformation on the t of flight of the diagram

        Parameters
        ----------
        trans
            Limited trasnformer to use
        hidden_size
            Hidden size for the RealMVP used as conditioner
        """
        super().__init__()

        dia_size = int(torch.sum(self.get_blocks_lenght()))

        self.__split = [dia_size - 1, 1]

        # Defines the transformation to swap the block in the right place
        self.__swap  = SwapDiaBlock("tm_fly", -1) 

        # Define the conditioner based on the split
        self.__cond  = RealMVP(self.__split[0], hidden_size, hidden_size, trans.trans_features)

        # Save transformer
        self.__trans = trans


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Apply the transformation to the diagrams and computes the log det of the transformation

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h  = self.__cond(z1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h)), dim=1))
        return  z, self.__trans.log_det(z2, h)


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the inverse transformation to the orders of the diagrams and computes the log det of the inverse

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        z2 = self.__trans.inverse(z2, h)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h)



class TOBCoupling(Flow, Diagrammatic):
    """
    Diagrammatic Coupling Flow specific for the time ordered blocks
    """
    def __init__(self, block: int | str, trans: Transformer, hidden_size: int = 100):
        """
        COnstructor

        Generates a Diagrammatic Coupling Flow that transforms the time ordered block and automatically computes the limits for the creation and anhilation times to insert inside the transformer.

        Parameters
        ----------
        block
            Time order block to which apply the transformation
        trans
            Transformer that maps R to [0, 1]
        hidden_size
            Hidden size of the RealMVP used as conditioner

        Raises
        ------
        KeyError:
            If the block selected is not a time ordered one
        """
        super().__init__()

        # Control the block passed
        if self.get_block_type(block) != block_types.tm_ordered:
            raise KeyError("Non time ordered block inserted inside time orfered coupling!")

        # Take the right index
        if isinstance(block, str):
            block = self.get_block(block)

        # Define the split list based on the diagram sizes
        dia_size = int(torch.sum(self.get_blocks_lenght()))
        n_couple = int(self.get_blocks_lenght()[block]) // 2
        even     = n_couple % 2

        self.__split = [dia_size - 2*(n_couple // 2 + even), 2*(n_couple // 2 + even)]

        # Defines the transformation to swap the block in the right place
        self.__swap = SwapDiaBlock(block, -1) 

        # Define the conditioner based on the split
        self.__cond = RealMVP(self.__split[0], hidden_size, hidden_size, self.__split[1] * trans.trans_features)

        # Save transformer
        self.__trans = trans

        # Retrive the time of flight
        self.__t = self.get_block("tm_fly")


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Apply the transformation to the selected block and computes the log determinant of the transformation

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det of the transformation
        """
        # Swap the eanted block at the end and split the diagram
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        tm_fly = self.get_block_from(self.__t, z)
        tc, td = (0.9999999 * tm_fly * self.__trans(z2, h)).chunk(2, dim=1)
        td = (1 - tc/tm_fly) * td + tc

        bad = torch.isinf(torch.log(tm_fly*(tm_fly - tc)).sum(dim=1))
        if bad.any():
            print(f"Esploso il logaritmo in TOBCoupling!")
            print(f"Parameters:\n{h[bad]}")
            print(f"Input:\n{z2[bad]}")
            print(f"Creation time:\n{tc[bad]}")
            print(f"Destruction time:\n{td[bad]}")
            print(f"Time of fly:\n{tm_fly[bad]}")

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, torch.cat((tc, td), dim=1)), dim=1)) 
        return z, torch.log(tm_fly*(tm_fly - tc)).sum(dim=1) + self.__trans.log_det(z2, h)


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the inverse to the selected block and computed the log det of the inverse transformation

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed batch and log det
        """
        # Swap the eanted block at the end and split the diagram
        z, _ = self.__swap(z)
        z1, z2 = z.split(self.__split, dim=1)
        h = self.__cond(z1)

        # First subtract the creation to destruction
        tm_fly = self.get_block_from(self.__t, z)
        tc, td = (z2 / tm_fly).chunk(2, dim=1)
        td = (td - tc) / (1 - tc)

        # Invert
        z2 = self.__trans.inverse(torch.cat((tc, td), dim=1), h)

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return z, -torch.log(tm_fly*tm_fly*(1 - tc)).sum(dim=1) - self.__trans.log_det(z2, h)
