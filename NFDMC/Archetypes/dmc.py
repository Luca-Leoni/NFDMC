import torch

from torch import Tensor
from enum import Enum

#--------------------------------

class block_types(Enum):
    """
    Enumerate class used in order to define the possible types of the blocks inside a diagram, the different diagramattic flows may need to act in different ways with different type of blocks.

    Attributes
    ----------
    integer
        The block contains integers value, normaly used for the block containing the order
    floating
        The block contains floating value, normaly used for block containing the time of flight of the particle
    tm_ordered
        The block contains a series of time ordered couples of creation and anhilation times
    """
    integer = "integer"
    floating = "floating"
    tm_ordered = "tm_ordered"

class Diagrammatic:
    """
    Archetipe class that is needed in order to define the diagram that we are going to use in the computation and track the position of the different types of variables as it gets transformed
    """
    __block_lenghts = torch.tensor([], dtype=torch.long)
    __block_types   = []
    __block_name    = dict()

    __dia_comp      = torch.tensor([], dtype=torch.long)


    def __init__(self):
        """
        Constructor

        Nothing happens here, the construction of the diagram needs to be done separatelly through static methods. In fact objects of this class should not be constructed or used and modifications should be possible only from Diagrammatic modules.
        """
        super().__init__()

    @staticmethod
    def add_block(name: str, lenght: int, type: block_types):
        """
        Add a block to the diagram definition

        Define a piece of the diagram by giving it a name and the lenght of the piece inside the vector representation of the diagram itself.

        Parameters
        ----------
        name
            Name of the block stored
        lenght
            Lenght of the block in the data vector
        type
            Type of the block inserted
        """
        Diagrammatic.__block_name[name] = len(Diagrammatic.__block_lenghts)
        Diagrammatic.__block_lenghts = torch.cat( (Diagrammatic.__block_lenghts, torch.tensor([int(lenght)], dtype=torch.long)))
        Diagrammatic.__block_types.append(type)

        Diagrammatic().__compute_dia_comp()



    @staticmethod
    def clear():
        """
        Clear the static internal variables that defines the composition, restart from zero.
        """
        Diagrammatic.__block_lenghts = torch.tensor([], dtype=torch.long)
        Diagrammatic.__block_types.clear()
        Diagrammatic.__block_name    = dict()
        Diagrammatic.__dia_comp      = torch.tensor([], dtype=torch.long)

    @staticmethod
    def print_comp():
        """
        Function to print the diagram composition to standard output in a, hopefully, preaty table format.
        """
        dia_comp = Diagrammatic().get_dia_comp()

        print("{:^10} | {:^10} | {:^10} | {:^10}".format("BLOCK NAME", "START", "END", "TYPE"))
        print("{:-^49}".format("-"))
        for i, name in enumerate(Diagrammatic.__block_name):
            print("{:<10} | {:<10} | {:<10} | {:^10}".format(name, dia_comp[i, 0], dia_comp[i, 1], Diagrammatic.__block_types[i].value))
        print("{:-^49}".format("-"))

 
    def get_dia_comp(self) -> Tensor:
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
        return Diagrammatic.__dia_comp

    def get_block_types(self) -> list[block_types]:
        """
        Gets the all list with the block types of the different blocks in order

        Returns
        -------
        list[block_types]
            List with the block types
        """
        return Diagrammatic.__block_types

    def get_block_type(self, block: str | int) -> block_types:
        """
        Gets the type of a wanted block

        Parameters
        ----------
        block
            Name of the interested block

        Returns
        -------
        block_types
            Type of the block
        """
        return Diagrammatic.__block_types[self.get_block(block) if isinstance(block, str) else block]


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
        if block1 != block2:
            pos1 = Diagrammatic.__block_name[block1]
            pos2 = Diagrammatic.__block_name[block2]
            lenghts = torch.clone(Diagrammatic.__block_lenghts)

            Diagrammatic.__block_lenghts[pos1], Diagrammatic.__block_lenghts[pos2] = lenghts[pos2], lenghts[pos1] 
            Diagrammatic.__block_name[block1], Diagrammatic.__block_name[block2] = Diagrammatic.__block_name[block2], Diagrammatic.__block_name[block1]

            self.__compute_dia_comp()


    def flip_dia_comp(self):
        """
        Flips the composition of the diagram due to a possible flip in the vector describing the diagrams
        """
        Diagrammatic.__block_lenghts = torch.flip(Diagrammatic.__block_lenghts, dims=(0,))

        size  = len(Diagrammatic.__block_name)
        items = Diagrammatic.__block_name.copy().items()

        for value in range(size):
            for name, pos in items:
                if pos == size - value - 1:
                    Diagrammatic.__block_name[name] = value

        self.__compute_dia_comp()


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


    def set_initial_comp(self):
        """
        Set back the composition to its original form basically eliminating all the block swaps that have been done so far, it's still not able to eliminate the swaps or permutations done internally inside the blocks.
        """
        Diagrammatic.__block_lenghts = Diagrammatic.__block_lenghts[list(Diagrammatic.__block_name.values())]

        for i, name in enumerate(Diagrammatic.__block_name):
            Diagrammatic.__block_name[name] = i

        self.__compute_dia_comp()


    def get_block_from(self, block: int | str, z: Tensor, bias_l: int = 0, step: int = 1) -> Tensor:
        """
        Utility function in order to retrive a wanted block from a batch of diagrams

        Parameters
        ----------
        block
            Block that you want to retrive, cna be the index of it or the name
        z
            Batch of diagrams
        bias_l
            bias to add to the left position of the bloc, sometimes is usefull
        step
            Step to take into the slicing of the array z[:, a:b:step]

        Returns
        -------
        Tensor
            Slice of the array that contains teh wanted block
        """
        if isinstance(block, str):
            block = self.get_block(block)

        return z[:, Diagrammatic.__dia_comp[block,0]+bias_l:Diagrammatic.__dia_comp[block,1]:step]

    def get_block_name(self, block: int) -> str:
        """
        Returns the name of the selected block inside the configuration.

        Parameters
        ----------
        block
            Index of the block in the configuration
        """
        return list(Diagrammatic.__block_name.keys())[block]

    
    def get_blocks_lenght(self) -> Tensor:
        """
        Returns the lenghts vector

        Returns
        -------
        Tensor
            Lenghts vector
        """
        return Diagrammatic.__block_lenghts


    def __compute_dia_comp(self):
        """
        Evaluates the diagram composition and stores it inside the static variable inside the class, needs to be called every time the composition gets modified.
        """
        Diagrammatic.__dia_comp = torch.zeros(Diagrammatic.__block_lenghts.shape[0], 2, dtype=torch.long)
        
        Diagrammatic.__dia_comp[0,1] = Diagrammatic.__block_lenghts[0]
        
        # Loop in order to set all others initial positions
        for i in range(1, Diagrammatic.__dia_comp.shape[0]):
            Diagrammatic.__dia_comp[i, 0] = Diagrammatic.__dia_comp[i-1, 1]
            Diagrammatic.__dia_comp[i, 1] = Diagrammatic.__dia_comp[i, 0] + Diagrammatic.__block_lenghts[i]

        Diagrammatic.__dia_comp = Diagrammatic.__dia_comp[list(Diagrammatic.__block_name.values()), :]
