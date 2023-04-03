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


class OBCoupling(Flow, Diagrammatic):
    def __init__(self, trans: Transformer, hidden_size: int = 100) -> None:
        super().__init__()

        dia_size = int(torch.sum(self.get_blocks_lenght()))

        self.__split = [dia_size - 1, 1]

        # Defines the transformation to swap the block in the right place
        self.__swap  = SwapDiaBlock("order", -1) 

        # Define the conditioner based on the split
        self.__cond  = RealMVP(self.__split[0], hidden_size, hidden_size, trans.trans_features-1)

        # Save transformer
        self.__trans = trans

        # Compute maximum possible order
        self.__max   = torch.tensor(0)
        for i, bt in enumerate(self.get_block_types()):
            if bt == block_types.tm_ordered:
                self.__max += self.get_blocks_lenght()[i] // 2

    
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h  = torch.cat((self.__cond(z1), torch.full((z.shape[0], 1), int(self.__max), device=z.device)), dim=1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h, self.__max)), dim=1))
        return  z, self.__trans.log_det(z2, h, self.__max)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        h  = torch.cat((self.__cond(z1), torch.full((z.shape[0], 1), int(self.__max), device=z.device)), dim=1)
        z2 = self.__trans.inverse(z2, h, self.__max)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h, self.__max)


class TFBCoupling(Flow, Diagrammatic):
    def __init__(self, trans: Transformer, max_tm: float = 50, hidden_size: int = 100) -> None:
        super().__init__()

        dia_size = int(torch.sum(self.get_blocks_lenght()))

        self.__split = [dia_size - 1, 1]

        # Defines the transformation to swap the block in the right place
        self.__swap  = SwapDiaBlock("tm_fly", -1) 

        # Define the conditioner based on the split
        self.__cond  = RealMVP(self.__split[0], hidden_size, hidden_size, trans.trans_features-1)

        # Save transformer
        self.__trans = trans

        # Search of time ordered blocks
        self.__tmb   = []
        for i, bt in enumerate(self.get_block_types()):
            if bt == block_types.tm_ordered:
                self.__tmb.append(i)


        self.__max   = torch.tensor(max_tm)


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        
        # search maximum
        max = torch.zeros(z.shape[0], device=z.device)
        for block in self.__tmb:
            max = torch.max(self.get_block_from(block, z), dim=1, keepdim=True)[0]

        
        h  = torch.cat((self.__cond(z1), (self.__max - max)), dim=1)
        z, _ = self.__swap(torch.cat((z1, self.__trans(z2, h, self.__max) + max), dim=1))

        bad = z[:, 1] > self.__max
        if bad.any():
            print(z2[bad])
            print(h[bad])
            print(max[bad])
            print(self.__trans(z2, h)[bad])


        return  z, self.__trans.log_det(z2, h, self.__max)


    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.__swap(z)
        z1, z2 = torch.split(z, self.__split, dim=1)
        
        # search maximum
        max = torch.zeros(z.shape[0], device=z.device)
        for block in self.__tmb:
            max = torch.max(self.get_block_from(block, z), dim=1, keepdim=True)[0]

        h  = torch.cat((self.__cond(z1), (self.__max - max)), dim=1)
        z2 = self.__trans.inverse(z2 - max, h, self.__max)
        z, _ = self.__swap(torch.cat((z1, z2), dim=1))
        return  z, -self.__trans.log_det(z2, h, self.__max)




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

        # Evaluate the new creation times (No idea why the clone is needed but it IS)
        tc = self.__trans(z2[:, ::2], h[:, 0::2, :].flatten(1).clone())

        # Update constrain on destruction to tm_fly - new tm_creation
        h[:, 1::2, self.__trans.trans_features-1] -= tc

        # Evaluate the new destruction times
        td = self.__trans(z2[:, 1::2], h[:, 1::2, :].flatten(1), self.get_block_from(self.__t, z)) + tc


        zf = torch.cat( (tc.reshape(tc.shape[0], tc.shape[1], 1), td.reshape(tc.shape[0], td.shape[1], 1)), dim=2 ).flatten(1)
        bad = (zf > z[:, 1:2]).any(dim=1)
        bad = bad | torch.isnan(zf).any(dim=1) | torch.isinf(zf).any(dim=1)
        if bad.any():
            print(f"Old tmd:\n{z2[bad]}")
            print(f"Old tmc:\n{z1[bad]}")
            print(f"New tmd\n{td[bad]}")
            print(f"New tmc:\n{tc[bad]}")
            print(f"Parameters:\n{h[bad]}")

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
        h = torch.cat((h, 0.99*self.get_block_from(self.__t, z).unsqueeze(1).expand(h.shape[0], h.shape[1], 1)), dim=2)

        # Set the constrains for destruction to tm_fly - tm_creation
        h[:, 1::2, self.__trans.trans_features-1] -= z2[:, ::2]
        h[:, 1::2, self.__trans.trans_features-1] *= 0.98

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
