import torch
import torch.nn as nn

from torch import Tensor
from ..Archetypes import Flow, Diagrammatic, block_types, Transformer
from ..Modules.nets import RealMVP
from .permutation import SwapDiaBlock

#------------------------------------------

class BCoupling(Flow, Diagrammatic):
    def __init__(self, block: int | str, trans: Transformer, hidden_size: int = 100):
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
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h)), dim=1))
        return  z, self.__trans.log_det(z2, h)


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h = self.__cond(z1)
        z2 = self.__trans.inverse(z2,h)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h)



class TOBCoupling(Flow, Diagrammatic):
    def __init__(self, block: int | str, trans: Transformer, hidden_size: int = 100):
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
        self.__cond = RealMVP(self.__split[0], hidden_size, hidden_size, self.__split[1] * (trans.trans_features-1))

        # Save transformer
        self.__trans = trans

        # Retrive the time of flight
        self.__t = self.get_block("tm_fly")


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # Swap the eanted block at the end and split the diagram
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)

        # Compute h and add constrain equal to tm_fly
        h = self.__cond(z1).reshape(z2.shape[0], z2.shape[1], self.__trans.trans_features-1)
        h = torch.cat((h, self.get_block_from(self.__t, z).unsqueeze(1).expand(h.shape[0], h.shape[1], 1)), dim=2)

        # Evaluate the new creation times
        tc  = self.__trans(z2[:, ::2], h[:,::2, :].flatten(1))

        # Update constrain on destruction to tm_fly - new tm_creation
        h[:, 1::2, self.__trans.trans_features-1] -= tc

        # Evaluate the new destruction times
        td = self.__trans(z2[:, 1::2], h[:, 1::2, :].flatten(1)) + tc

        # Reconstruct couples
        z2 = torch.cat( (tc.reshape(tc.shape[0], tc.shape[1], 1), td.reshape(tc.shape[0], td.shape[1], 1)), dim=2 ).flatten(1)

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))

        return z, self.__trans.log_det(z2, h.flatten(1))


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # Swap the eanted block at the end and split the diagram
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)

        # Compute h and add constrain equal to tm_fly
        h = self.__cond(z1).reshape(z2.shape[0], z2.shape[1], self.__trans.trans_features-1)
        h = torch.cat((h, self.get_block_from(self.__t, z).unsqueeze(1).expand(h.shape[0], h.shape[1], 1)), dim=2)

        # Set the constrains for destruction to tm_fly - tm_creation
        h[:, 1::2, self.__trans.trans_features-1] -= z2[:, ::2]

        # First subtract the creation to destruction
        tc = z2[:, ::2]
        td = z2[:, 1::2] - tc

        # Reconstruct couples
        z2 = torch.cat( (tc.reshape(tc.shape[0], tc.shape[1], 1), td.reshape(tc.shape[0], td.shape[1], 1)), dim=2 ).flatten(1)

        # Invert
        z2 = self.__trans.inverse(z2, h.flatten(1))

        # Recreate the diagrams batch
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))

        return z, -self.__trans.log_det(z2, h.flatten(1))
