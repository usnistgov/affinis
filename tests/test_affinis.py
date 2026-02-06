import numpy as np
from hypothesis import given, settings, example
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import pytest

from affinis import associations as asc

# def test_version():
#     assert __version__ == "0.1.0"

@st.composite  
def make_shapes(draw):
    return draw(hnp.array_shapes(max_dims=2, min_dims=2, min_side=2))


@st.composite
def make_bools(draw, shape=(2,5)): 
    arr = draw(hnp.arrays(
        bool,
        shape,
        elements=st.just(True),
        fill=st.just(False),
    ))
    return arr


# @example((np.array([[0,1,0,1],[0,1,1,0]]))).xfail() #ints aren't bools
@pytest.mark.parametrize(
    "assoc_func",
    [
        asc.binary_cosine_similarity,
        asc.coocur_prob,
        asc.ochiai,
        asc.high_salience_skeleton,
        asc.mutual_information,
        asc.odds_ratio,
        asc.yule_q,
        asc.yule_y,
        asc.forest_pursuit,
        asc.forest_pursuit_cts,
        asc.forest_pursuit_interaction,
        asc.expected_forest_maximization,
        asc.forest_pursuit_edge,
        asc.chow_liu,
        asc.resource_project
    ]
                                  
)
@given(data=st.data())
def test_numpy_shapes(assoc_func, data):
    X = data.draw(make_bools())
    D = assoc_func(X)
    assert D.shape == (X.shape[-1], X.shape[-1])



