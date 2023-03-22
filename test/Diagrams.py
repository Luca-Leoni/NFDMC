import torch

from NFDMC.Archetypes import Diagrammatic
from NFDMC.Distributions import diagrams
from hypothesis import given, settings, strategies as st

@given(st.integers(min_value=2, max_value=100).filter(lambda x: not x % 2))
def test_Holstein(max_order: int):
    dis = diagrams.Holstein(1, 0.2, 0.5)

    z = torch.rand(2, max_order+1)
    z[:, 0] = torch.randint(low=0, high=max_order, size=(2,))

    # Construct a diagram
    for i in range(1, max_order, 2):
        z[0, i+1] += z[0, i]

    # Set second as non diagram
    z[1, 2] = 0

    log_p = dis.log_prob(z)

    assert log_p.shape == (2,)
    assert log_p[1] == -1000000
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
        Diagrammatic.add_block(f"block{i}", lenghts[i])

    dia_comp = Diagrammatic.get_dia_comp()
    Diagrammatic.clear()

    assert (dia_comp == comp).all()


def test_diagrammatic_swap():
    lenghts = [1, 100, 30, 12]

    for i in range(len(lenghts)):
        Diagrammatic.add_block(f"block{i}", lenghts[i])


    Diagrammatic().swap_blocks(f"block0", f"block2")

    print(Diagrammatic.get_dia_comp())

    result = torch.tensor([[130, 131],
                           [30, 130],
                           [0, 30],
                           [131, 143]])

    assert (Diagrammatic.get_dia_comp() == result).all()

    Diagrammatic().swap_blocks("block1", "block0")

    result = torch.tensor([[30, 31],
                           [31, 131],
                           [0, 30],
                           [131, 143]])

    assert (Diagrammatic.get_dia_comp() == result).all()

if __name__ == '__main__':
    test_diagrammatic_initialization()
