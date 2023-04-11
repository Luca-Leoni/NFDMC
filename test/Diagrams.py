from os import lchown
import torch
import numpy as np

from NFDMC.Archetypes import Diagrammatic, block_types
from NFDMC.Distributions import diagrams
from NFDMC.Flows.permutation import DBsFlip, SwapDiaBlock, PermuteTimeBlock, DBFlip
from NFDMC.Flows.dmc import BCoupling, TOBCoupling
from NFDMC.Flows import transformer
from hypothesis import given, settings, strategies as st

# torch.autograd.set_detect_anomaly(True)

#---------------------------------------------

def check_if_a_in_b(a, b) -> bool:
    present = False
    for element in b:
        if (a == element).all():
            present = True

    return present


def create_random_diagram(n_block: int, btype: block_types) -> np.ndarray:
    Diagrammatic.clear()

    lenghts = np.random.randint(low=1, high=10, size=(n_block,))
    if btype == block_types.tm_ordered:
        lenghts *= 2

    for i, lenght in enumerate(lenghts):
        Diagrammatic.add_block(f"block{i}", lenght, btype)

    return lenghts

#---------------------------------------------

@given(st.integers(min_value=2, max_value=100).filter(lambda x: not x % 2))
def test_Holstein(max_order: int):
    Diagrammatic.clear()
    Diagrammatic.add_block("order", 1, block_types.integer)
    Diagrammatic.add_block("tm_fly", 1, block_types.floating)
    Diagrammatic.add_block("phonons", max_order, block_types.tm_ordered)

    dis = diagrams.Holstein(1, 0.2, 0.5)

    z = torch.rand(2, max_order+2, dtype=torch.float64)
    z[:, 0] = torch.randint(low=0, high=max_order, size=(2,))

    # Construct a diagram
    for i in range(2, max_order, 2):
        z[0, i+1] += z[0, i]

    # Set second as non diagram
    z[1, -1] = 0

    log_p = dis.log_prob(z)

    assert log_p.shape == (2,)


@given(n_blocks=st.integers(min_value=2, max_value=10))
def test_diagrammatic_initialization(n_blocks):
    Diagrammatic.clear()
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


@settings(deadline=5000)
@given(batch=st.integers(min_value=1, max_value=100),
       block=st.integers(min_value=0, max_value=4))
def test_Softplus_BCoupling(batch, block: int):
    Diagrammatic.clear()

    lenghts = torch.randint(low=1, high=10, size=(5,)).data.numpy()
    bias = lenghts[block] // 2 + lenghts[block] % 2
    beg = np.sum(lenghts[:block]) + bias
    end = beg + lenghts[block] - bias

    for i in range(5):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.floating)
    diagrams = torch.rand(batch, np.sum(lenghts)).to("cuda", dtype=torch.float64)

    flow = BCoupling(block, transformer.Softplus()).to("cuda")

    transformed, log_det = flow(diagrams)
    inversed, log_det_in = flow.inverse(transformed)

    # Present only to see if the backward is possible
    torch.sum(log_det).backward()

    assert torch.isclose(log_det, -log_det_in, rtol=0., atol=1e-5).all()
    assert torch.isclose(diagrams, inversed, rtol=0., atol=1e-5).all()
    assert (diagrams[:, beg:end] != transformed[:, beg:end]).all()
    if block != 4:
        assert (diagrams[:, np.sum(lenghts[:-1]):] == transformed[:, np.sum(lenghts[:-1]):]).all()

@settings(deadline=5000)
@given(batch=st.integers(min_value=1, max_value=100),
       block=st.integers(min_value=0, max_value=4))
def test_Sigmoid_BCoupling(batch, block: int):
    Diagrammatic.clear()

    lenghts = torch.randint(low=1, high=10, size=(5,)).data.numpy()
    bias = lenghts[block] // 2 + lenghts[block] % 2
    beg = np.sum(lenghts[:block]) + bias
    end = beg + lenghts[block] - bias

    for i in range(5):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.floating)
    diagrams = torch.rand(batch, np.sum(lenghts), dtype=torch.float64).to("cuda")

    flow = BCoupling(block, transformer.Sigmoid()).to("cuda")

    transformed, log_det = flow(diagrams)
    inversed, log_det_in = flow.inverse(transformed)

    # Present only to see if the backward is possible
    torch.sum(log_det).backward()

    assert torch.isclose(log_det, -log_det_in, rtol=0., atol=1e-5).all()
    assert torch.isclose(diagrams, inversed, rtol=0., atol=1e-5).all()
    assert (diagrams[:, beg:end] != transformed[:, beg:end]).all()
    if block != 4:
        assert (diagrams[:, np.sum(lenghts[:-1]):] == transformed[:, np.sum(lenghts[:-1]):]).all()


@settings(deadline=5000)
@given(batch=st.integers(min_value=1, max_value=100),
       block=st.integers(min_value=0, max_value=4))
def test_Sigmoid_TOBCoupling(batch, block: int):
    Diagrammatic.clear()

    Diagrammatic.add_block("tm_fly", 1, block_types.floating)

    lenghts = torch.randint(low=1, high=5, size=(5,)).data.numpy()*2
    n_couples = lenghts[block] // 2
    bias = 2*(n_couples // 2 + n_couples % 2)
    beg = np.sum(lenghts[:block]) + bias + 1
    end = beg + lenghts[block] - bias

    for i in range(5):
        Diagrammatic.add_block(f"block{i}", lenghts[i], block_types.tm_ordered)
    diagrams = torch.rand(batch, np.sum(lenghts)+1, dtype=torch.float64).to("cuda")

    # Set tm_fly greater than all other otherwise CPAffine does not work
    diagrams[:, 0] += 1

    flow = TOBCoupling(block+1, transformer.Sigmoid()).to("cuda")

    # print("Forward")
    transformed, log_det = flow(diagrams)
    # print("Inverse")
    inversed, log_det_in = flow.inverse(transformed) 
    # print()

    # Present only to see if the backward is possible
    torch.sum(log_det).backward()

    assert torch.isclose(log_det, -log_det_in, rtol=0., atol=1e-5).all()
    assert torch.isclose(diagrams, inversed, rtol=0., atol=1e-5).all()
    assert (diagrams[:, beg:end] != transformed[:, beg:end]).all()
    if block != 4:
        assert (diagrams[:, np.sum(lenghts[:-1])+1:] == transformed[:, np.sum(lenghts[:-1])+1:]).all()

@settings(deadline=5000)
@given(batch=st.integers(min_value=1, max_value=100),
       n_block=st.integers(min_value=1, max_value=5))
def test_DBFlip(batch, n_block):
    lenghts  = create_random_diagram(n_block, block_types.floating) 
    diagrams = torch.rand(size=(batch, np.sum(lenghts)))

    block    = int(np.random.randint(low=0, high=n_block, size=(1,))) 
    permute  = DBFlip(block)

    permuted, _ = permute(diagrams)

    beg = np.sum(lenghts[:block])
    end = beg + lenghts[block]
    assert (diagrams[:, :beg] == permuted[:, :beg]).all()
    assert (diagrams[:, end:] == permuted[:, end:]).all()
    assert (permuted[:, beg:end] == torch.flip(diagrams[:, beg:end], dims=(1,))).all()


@settings(deadline=5000)
@given(batch=st.integers(min_value=1, max_value=100),
       n_block=st.integers(min_value=1, max_value=5))
def test_DBFlip_tm_ordered(batch, n_block):
    lenghts  = create_random_diagram(n_block, block_types.tm_ordered) 
    diagrams = torch.rand(size=(batch, np.sum(lenghts)))

    block    = int(np.random.randint(low=0, high=n_block, size=(1,))) 
    permute  = DBFlip(block)

    permuted, _ = permute(diagrams)

    beg = np.sum(lenghts[:block])
    end = beg + lenghts[block]
    assert (diagrams[:, :beg] == permuted[:, :beg]).all()
    assert (diagrams[:, end:] == permuted[:, end:]).all()
    assert (permuted[:, beg:end] == torch.flip(diagrams[:, beg:end].reshape(batch, lenghts[block] // 2, 2), dims=(1,)).flatten(1)).all()


@settings(deadline=5000)
@given(batch=st.integers(min_value=1000, max_value=20000),
        max_order=st.integers(min_value=2, max_value=100).filter(lambda x: not x % 2))
def test_BaseHolstein(batch, max_order):
    Diagrammatic.clear()
    Diagrammatic.add_block("order", 1, block_types.integer)
    Diagrammatic.add_block("tm_fly", 1, block_types.floating)
    Diagrammatic.add_block("phonons", max_order, block_types.tm_ordered)

    dis = diagrams.BaseHolstein(max_order)

    dia, log_p = dis(batch)
    tc, td     = dia[:, 2:].chunk(2, dim=1)

    assert log_p.shape == (batch,)
    assert dia.shape == (batch, max_order + 2)
    assert (dia[:, 0:2] > 0).all()
    assert not (dia.isinf() | dia.isnan()).any()
    assert not (log_p.isnan() | log_p.isinf()).any()
    assert (td < dia[:, 1:2]).all()
    assert (td > tc).all()

#--------------------------------------

if __name__ == '__main__':
    test_diagrammatic_initialization()
