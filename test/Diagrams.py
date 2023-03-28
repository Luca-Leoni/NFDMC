import torch
import numpy as np

from NFDMC.Archetypes import Diagrammatic, block_types
from NFDMC.Distributions import diagrams
from NFDMC.Flows.permutation import SwapDiaBlock, PermuteTimeBlock
from NFDMC.Flows.dmc import DiaChecker
from hypothesis import given, settings, strategies as st

#---------------------------------------------

def check_if_a_in_b(a, b) -> bool:
    present = False
    for element in b:
        if (a == element).all():
            present = True

    return present

#---------------------------------------------

@given(st.integers(min_value=2, max_value=100).filter(lambda x: not x % 2))
def test_Holstein(max_order: int):
    dis = diagrams.Holstein(1, 0.2, 0.5)

    z = torch.rand(2, max_order+2)
    z[:, 0] = torch.randint(low=0, high=max_order, size=(2,))

    # Construct a diagram
    for i in range(2, max_order, 2):
        z[0, i+1] += z[0, i]

    # Set second as non diagram
    z[1, -1] = 0

    log_p = dis.log_prob(z)

    assert log_p.shape == (2,)
    assert log_p[1] <= -100
    assert log_p[0] != -1000000


@given(n_blocks=st.integers(min_value=2, max_value=10))
def test_diagrammatic_initialization(n_blocks):
    comp = torch.randint(low=1, high=100, size=(n_blocks, 2))
    lenghts = torch.clone(comp[:, 1]).data.numpy()

    comp[0,0] = 0
    for i in range(1, n_blocks):
        comp[i,0] = comp[i-1, 1]
        comp[i,1] = comp[i,0] + lenghts[i]

    for i in range(n_blocks):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)

    dia_comp = Diagrammatic().get_dia_comp()
    Diagrammatic.clear()

    assert (dia_comp == comp).all()


def test_diagrammatic_swap():
    lenghts = [1, 100, 30, 12]

    for i in range(len(lenghts)):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)


    Diagrammatic().swap_blocks(f"block0", f"block2")

    print(Diagrammatic().get_dia_comp())

    result = torch.tensor([[130, 131],
                           [30, 130],
                           [0, 30],
                           [131, 143]])

    assert (Diagrammatic().get_dia_comp() == result).all()

    Diagrammatic().swap_blocks("block1", "block0")

    result = torch.tensor([[30, 31],
                           [31, 131],
                           [0, 30],
                           [131, 143]])

    assert (Diagrammatic().get_dia_comp() == result).all()


@given(batch=st.integers(min_value=1, max_value=10),
        block=st.integers(min_value=0, max_value=9))
def test_time_permutation(batch, block):
    Diagrammatic.clear()

    # Generate composition
    lenghts = torch.randint(low=2, high=50, size=(10,)).data.numpy() * 2
    for i in range(10):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)

    diagrams = torch.arange(np.sum(lenghts)*batch).reshape(batch, np.sum(lenghts))

    #Do the thing
    permute = PermuteTimeBlock(f"block{block}").to("cuda")
 
    permuted, _ = permute(diagrams)
    inverted, _ = permute.inverse(permute(diagrams)[0])

    start = np.sum(lenghts[:block])
    end   = start + lenghts[block]
    per_block = diagrams[:, start:end].reshape(batch, lenghts[block] // 2, 2) 

    permuted = permuted[:, start:end]
    permuted = permuted.reshape(batch, lenghts[block] // 2, 2)

    for i, couples in enumerate(per_block):
        for couple in couples:
            assert check_if_a_in_b(couple, permuted[i])
    assert (inverted == diagrams).all()


@given(batch=st.integers(min_value=1, max_value=10),
        block1=st.integers(min_value=0, max_value=9),
        block2=st.integers(min_value=0, max_value=9))
def test_time_swap(batch, block1, block2):
    if block1 == block2:
        return

    Diagrammatic.clear()

    # I want block1 to be before the two for simplicity in terms of testing
    if block1 > block2:
        block1, block2 = block2, block1

    # Generate composition
    lenghts = torch.randint(low=2, high=10, size=(10,)).data.numpy() * 2
    for i in range(10):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)

    diagrams = torch.arange(np.sum(lenghts)*batch).reshape(batch, np.sum(lenghts))

    # Do the thing
    swap = SwapDiaBlock(f"block{block1}", f"block{block2}")

    permuted, _ = swap(diagrams)
    inverted, _ = swap.inverse(torch.clone(permuted))
    
    start1 = np.sum(lenghts[:block1])
    start2 = np.sum(lenghts[:block2])
    end1 = start1 + lenghts[block1]
    end2 = start2 + lenghts[block2]

    b1 = diagrams[:, start1:end1]
    b2 = diagrams[:, start2:end2]

    assert (b2 == permuted[:, start1:start1 + lenghts[block2]]).all()
    assert (b1 == permuted[:, start2 - lenghts[block1] + lenghts[block2]:start2 + lenghts[block2]]).all()
    assert (inverted == diagrams).all()


@given(batch=st.integers(min_value=1, max_value=10),
        block1=st.integers(min_value=0, max_value=9),
        block2=st.integers(min_value=0, max_value=9))
def test_time_swap_permute(batch, block1, block2):
    if block1 == block2:
        return

    Diagrammatic.clear()

    # I want block1 to be before the two for simplicity in terms of testing
    if block1 > block2:
        block1, block2 = block2, block1

    # Generate composition
    lenghts = torch.randint(low=2, high=10, size=(10,)).data.numpy() * 2
    for i in range(10):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)

    diagrams = torch.arange(np.sum(lenghts)*batch).reshape(batch, np.sum(lenghts))

    # Do the thing
    swap = SwapDiaBlock(f"block{block1}", f"block{block2}")
    permute = PermuteTimeBlock(f"block{block2}")

    permuted, _ = permute(swap(diagrams)[0])
    inverted, _ = permute.inverse(swap.inverse(torch.clone(permuted))[0])

    start = np.sum(lenghts[:block2])
    end   = start + lenghts[block2]
    per_block = diagrams[:, start:end].reshape(batch, lenghts[block2] // 2, 2)

    start = np.sum(lenghts[:block1])
    end   = start + lenghts[block2]
    permuted = permuted[:, start:end]
    permuted = permuted.reshape(batch, lenghts[block2] // 2, 2)
    
    for i, couples in enumerate(per_block):
        for couple in couples:
            assert check_if_a_in_b(couple, permuted[i])
    assert (inverted == diagrams).all()


@given(batch=st.integers(min_value=1, max_value=100),
       block1=st.integers(min_value=0, max_value=3),
       block2=st.integers(min_value=0, max_value=3))
def test_dia_checker(batch, block1, block2):
    Diagrammatic.clear()

    lenghts = torch.randint(low=1, high=10, size=(4,), dtype=torch.long).data.numpy() * 2
    types   = [block_types.integer, block_types.floating, block_types.tm_ordered, block_types.floating]

    for i in range(4):
        Diagrammatic.add_block(f"block{i}", int(lenghts[i]), types[i])

    z = torch.rand(size=(batch, int(np.sum(lenghts)))) * 3

    dia_check = DiaChecker()

    diagrams, _ = SwapDiaBlock(f"block{block1}", f"block{block2}")(z)
    diagrams, _ = dia_check(z)

    dia_comp = Diagrammatic().get_dia_comp()
    # assert (diagrams[:, dia_comp[0,0]:dia_comp[0,1]] == torch.floor(diagrams[:, dia_comp[0,0]:dia_comp[0,1]])).all()
    assert (diagrams[:, dia_comp[0,0]:dia_comp[0,1]] == z[:, dia_comp[0,0]:dia_comp[0,1]]).all()
    assert (diagrams[:, dia_comp[1,0]:dia_comp[1,1]] == z[:, dia_comp[1,0]:dia_comp[1,1]]).all()
    assert (diagrams[:, dia_comp[2,0]:dia_comp[2,1]:2] < diagrams[:, dia_comp[2,0]+1:dia_comp[2,1]:2]).all()


@given(batch=st.integers(min_value=1, max_value=100),
       block1=st.integers(min_value=0, max_value=3),
       block2=st.integers(min_value=0, max_value=3))
def test_dia_checker_last(batch, block1, block2):
    Diagrammatic.clear()

    lenghts = torch.randint(low=1, high=10, size=(4,), dtype=torch.long).data.numpy() * 2
    types   = [block_types.integer, block_types.floating, block_types.tm_ordered, block_types.floating]

    for i in range(4):
        Diagrammatic.add_block(f"block{i}", int(lenghts[i]), types[i])
    init_comp = Diagrammatic().get_dia_comp()

    z = torch.rand(size=(batch, int(np.sum(lenghts)))) * 3


    dia_check = DiaChecker(last=True)

    diagrams, _ = SwapDiaBlock(f"block{block1}", f"block{block2}")(z)
    diagrams, _ = dia_check(diagrams)


    dia_comp = Diagrammatic().get_dia_comp()
    assert (init_comp == dia_comp).all()
    assert (diagrams[:, dia_comp[0,0]:dia_comp[0,1]] == torch.floor(diagrams[:, dia_comp[0,0]:dia_comp[0,1]])).all()
    assert (diagrams[:, dia_comp[1,0]:dia_comp[1,1]] == z[:, dia_comp[1,0]:dia_comp[1,1]]).all()
    assert (diagrams[:, dia_comp[2,0]:dia_comp[2,1]:2] < diagrams[:, dia_comp[2,0]+1:dia_comp[2,1]:2]).all()


@given(batch=st.integers(min_value=1, max_value=100),
       n_blocks=st.integers(min_value=2, max_value=6))
def test_dia_flip(batch, n_blocks):
    Diagrammatic.clear()

    lenghts = torch.randint(low=3, high=10, size=(n_blocks,)).data.numpy()
    for i in range(n_blocks):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.integer)
    diagrams = torch.rand(batch, np.sum(lenghts))
    end_part = torch.flip(diagrams[:, np.sum(lenghts[:-1]):], dims=(1,))

    
    # flip everithing
    Diagrammatic().flip_dia_comp()
    dia_comp = Diagrammatic().get_dia_comp()
    fldia    = torch.flip(diagrams, dims=(1,))
    beg_part = fldia[:, dia_comp[-1, 0]:dia_comp[-1, 1]]

    # check
    assert (beg_part == end_part).all()

#--------------------------------------

if __name__ == '__main__':
    test_diagrammatic_initialization()
