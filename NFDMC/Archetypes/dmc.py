import torch

from torch import Tensor

class Diagrammatic:
    """
    Archetipe class that is needed in order to define the diagram that we are going to use in the computation and track the position of the different types of variables as it gets transformed
    """
    __block_lenghts = torch.tensor([])
    __block_name    = dict()

    def __init__(self):
        """
        COnstructor

        Nothing happens here, the construction of the diagram needs to be done separatelly through static methods. In fact objects of this class should not be constructed or used and modifications should be possible only from Diagrammatic modules.
        """
        super().__init__()

    @staticmethod
    def add_block(name: str, lenght: int):
        """
        Add a block to the diagram definition

        Define a piece of the diagram by giving it a name and the lenght of the piece inside the vector representation of the diagram itself.

        Parameters
        ----------
        name
            Name of the block stored
        lenght
            Lenght of the block in the data vector
        """
        Diagrammatic.__block_name[name] = len(Diagrammatic.__block_lenghts)
        Diagrammatic.__block_lenghts = torch.cat( (Diagrammatic.__block_lenghts, torch.tensor([lenght])) )



    @staticmethod
    def clear():
        """
        Clear the static internal variables that defines the composition, restart from zero.
        """
        Diagrammatic.__block_lenghts = torch.tensor([])
        Diagrammatic.__block_name    = dict()

    
    @staticmethod
    def get_dia_comp() -> Tensor:
        """
        Computes the diagram composition in that moment generating a 2D array composed in the following way:
            [[beg_idx_1, end_idx_1],
             ...
             [beg_idx_n, end_idx_n]]
        So that every row is relative to the first block added, the second to the second etc, and has two entries defining the indexes of the array representation of the diagram in which that block is present in the moment of the computation. So if I have added "order" and "phonon_time" as block to the diagram and along the flow I modify their relative positions trough permutations then I can find out where the order is by taking get_dia_comp()[0] and have the indexes that locate it.

        Returns
        -------
        Tensor
            2D array containing informations of begin and end of the different blocks
        """
        dia_comp = torch.zeros(Diagrammatic.__block_lenghts.shape[0], 2)
        
        dia_comp[0,1] = Diagrammatic.__block_lenghts[0]
        
        # Loop in order to set all others initial positions
        for i in range(1, dia_comp.shape[0]):
            dia_comp[i, 0] = dia_comp[i-1, 1]
            dia_comp[i, 1] = dia_comp[i, 0] + Diagrammatic.__block_lenghts[i]

        return dia_comp[list(Diagrammatic.__block_name.values()), :]


    def swap_blocks(self, block1: str, block2: str):
        """
        Swap position of two blocks in the composition

        Basically you tell the names of the two added blocks that you want to permute

        Parameters
        ----------
        block1
            name of the first block
        block2
            name of the second block
        """
        pos1 = Diagrammatic.__block_name[block1]
        pos2 = Diagrammatic.__block_name[block2]
        lenghts = torch.clone(Diagrammatic.__block_lenghts)

        Diagrammatic.__block_lenghts[pos1], Diagrammatic.__block_lenghts[pos2] = lenghts[pos2], lenghts[pos1] 
        Diagrammatic.__block_name[block1], Diagrammatic.__block_name[block2] = Diagrammatic.__block_name[block2], Diagrammatic.__block_name[block1]


    def get_block_pos(self, name: str) -> int:
        """
        Gives the block position relative to the others in the array in that moment, so that if the block "order" started as first, idx 0, and was permuted with the one inserted as third, "tm_fly", than now "order" will return 2 and "tm_fly" 0.

        Parameters
        ----------
        name
            Name of the block wanted

        Returns
        -------
        int
            Relative position of the block

        Raises
        ------
        KeyError:
            If the block name inserted is not defined
        """
        if name not in Diagrammatic.__block_name:
            raise KeyError(f"Block {name} was not defined!")

        return Diagrammatic.__block_name[name]

    def get_block(self, name: str) -> int:
        """
        Gets the original position of the block

        Parameters
        ----------
        name
            Name of the block wanted

        Returns
        -------
        int
            Original placement of the block

        Raises
        ------
        KeyError:
            If the name of the block is not defined
        RuntimeError:
            In case some strange bug would make the loop exit without answer
        """
        if name not in Diagrammatic.__block_name:
            raise KeyError(f"Block {name} was not defined!")

        for i, key in enumerate(Diagrammatic.__block_name):
            if key == name:
                return i

        raise RuntimeError("Something bad has happened :(")
