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
    def __init__(self, trans: LTransformer, hidden_size: int = 100) -> None:
        """
        COnstructor

        Generates a Diagrammatic Coupling flow that acts on the order of the diagram specifically. It automatically computes the upper limit for the order

        Parameters
        ----------
        trans
            Limited transformer to use
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

        # Set the transformer limit
        max   = torch.tensor(0.)
        for i, bt in enumerate(self.get_block_types()):
            if bt == block_types.tm_ordered:
                max += self.get_blocks_lenght()[i] // 2
        self.__trans.set_upper_limit(max)
        
        # For debugging
        self.__max = max

    
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

        bad = (z[:, 0] > self.__max)
        if bad.any():
            print("From OBCoupling:\n")
            print(z2[bad])
            print(h[bad])
            print(self.__trans(z2, h)[bad])

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
    def __init__(self, trans: LTransformer, max_tm: float = 50, hidden_size: int = 100) -> None:
        """
        Constructor

        Craetes a Diagrammatic Flow that apply the selected transformation on the t of flight of the diagram

        Parameters
        ----------
        trans
            Limited trasnformer to use
        max_tm
            Maximum limit for the time of flight
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

        # Search of time ordered blocks
        self.__tmb   = []
        for i, bt in enumerate(self.get_block_types()):
            if bt == block_types.tm_ordered:
                self.__tmb.append(i)

        # Save the maximum value
        self.__max = float(max_tm)


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
        
        # search maximum to set limit
        max = torch.zeros(z.shape[0], device=z.device)
        for block in self.__tmb:
            max = torch.max(self.get_block_from(block, z), dim=1, keepdim=True)[0]
        self.__set_limit(max)

        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h) + max), dim=1))

        bad = (z > self.__max).any(dim=1)
        if bad.any():
            print("From TFBCoupling:\n")
            print(z2[bad])
            print(h[bad])
            print(max[bad])
            print(self.__trans(z2, h)[bad])

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
        
        # search maximum
        max = torch.zeros(z.shape[0], device=z.device)
        for block in self.__tmb:
            max = torch.max(self.get_block_from(block, z), dim=1, keepdim=True)[0]
        self.__set_limit(max)

        z2 = self.__trans.inverse(z2 - max, h)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h)


    def __set_limit(self, max: Tensor):
        """
        Utility to set the limits of the transformer

        The limits are given by C = maximum_tm_fly - maximum_phonon_time and L = maximum_tm_fly for every element in the batch

        Parameters
        ----------
        max
            Array with all the maximum phonon times in every diagram
        """
        UL = torch.full((max.shape[0], 2), self.__max, device=max.device)
        UL[:, 1:2] -= max
        self.__trans.set_upper_limit(UL)




class TOBCoupling(Flow, Diagrammatic):
    """
    Diagrammatic Coupling Flow specific for the time ordered blocks
    """
    def __init__(self, block: int | str, trans: LTransformer, hidden_size: int = 100):
        """
        COnstructor

        Generates a Diagrammatic Coupling Flow that transforms the time ordered block and automatically computes the limits for the creation and anhilation times to insert inside the transformer.

        Parameters
        ----------
        block
            Time order block to which apply the transformation
        trans
            Limited transformation to use
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
        h = self.__cond(z1).reshape(z.shape[0], z2.shape[1], self.__trans.trans_features)

        tm_fly = self.get_block_from("tm_fly", z).clone()

        self.__set_limits(z)

        # Evaluate the new creation times (No idea why the clone is needed but it IS)
        tc = self.__trans(z2[:, ::2], h[:, 0::2, :].flatten(1))

        # Update constrain on destruction to tm_fly - new tm_creation
        self.__set_limits(z, tc)

        # Evaluate the new destruction times
        td = self.__trans(z2[:, 1::2], h[:, 1::2, :].flatten(1)) + tc

        zf = torch.cat( (tc.unsqueeze(-1), td.unsqueeze(-1)), dim=2 ).flatten(1)
        bad = (zf > z[:, 1:2]).any(dim=1) | (td < tc).any(dim=1)
        bad = bad | torch.isnan(zf).any(dim=1) | torch.isinf(zf).any(dim=1)
        if bad.any():
            print("FromTOBCoupling:\n")
            print(f"Old tmd:\n{z2[bad]}")
            print(f"Old tmc:\n{z1[bad]}")
            print(f"New tmd\n{td[bad]}")
            print(f"New tmc:\n{tc[bad]}")
            print(f"Parameters:\n{h[bad]}")

        # Reconstruct couples
        z2 = torch.cat( (tc.unsqueeze(-1), td.unsqueeze(-1)), dim=2 ).flatten(1)

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, z2), dim=1)) 

        if (tm_fly != self.get_block_from("tm_fly", z)).any():
            print("TOB fa casino!")

        self.__set_limits(z, tc, total=True)
        return z, self.__trans.log_det(z2, h.flatten(1))


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
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        
        # First subtract the creation to destruction
        tc = z2[:, ::2]
        td = z2[:, 1::2] - tc

        # Reconstruct couples
        z2 = torch.cat( (tc.reshape(tc.shape[0], tc.shape[1], 1), td.reshape(tc.shape[0], td.shape[1], 1)), dim=2 ).flatten(1)

        # Invert
        self.__set_limits(z, tc, total=True)
        z2 = self.__trans.inverse(z2, h)

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))

        return z, -self.__trans.log_det(z2, h)


    def __set_limits(self, z: Tensor, tc: Tensor | None = None, total: bool = False):
        if total:
            UL = self.get_block_from(self.__t, z).repeat(1, self.__split[1]*2)
            UL = UL.reshape(UL.shape[0], 2, self.__split[1])
            UL[:, 1, 1::2] -= tc
            self.__trans.set_upper_limit(UL)
        elif isinstance(tc, type(None)):
            self.__trans.set_upper_limit(self.get_block_from(self.__t, z))
        else:
            UL = self.get_block_from(self.__t, z).repeat(1, self.__split[1])
            UL[:, self.__split[1]//2 :] -= tc
            UL = UL.reshape(UL.shape[0], 2, self.__split[1] // 2)
            self.__trans.set_upper_limit(UL)
