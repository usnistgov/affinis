import numpy as np
from math import comb
from hypothesis import given, settings, example
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import pytest

from affinis.priors import pseudocount
from affinis.types import PsdCts

@st.composite
def make_vecs(draw, shape=6):
    return draw(hnp.arrays(
        float,shape,fill=st.just(0)
    ))

@st.composite
def make_num_den(draw, tri=False):
    if tri:
        shape=draw(st.integers(min_value=3, max_value=20))
        shape=comb(shape,2)
        print(shape)
    else:
        shape=draw(st.integers(min_value=2, max_value=20))
    num = draw(make_vecs(shape))
    den = draw(make_vecs(shape)) + num
    return num, den
    


@given(vec_pair=make_num_den(),a=st.floats(min_value=0.))
def test_pseudoct_symmetric_(vec_pair,a):
    x1,x2 = vec_pair
    psdct = pseudocount(a)
    psdct(x1,x2)


@given(
    vec_pair=make_num_den(),
    a=st.floats(min_value=0.),
    b=st.floats(min_value=0.)
)
def test_pseudoct_asymmetric_(vec_pair,a,b):
    x1,x2 = vec_pair
    psdct = pseudocount((a,b))
    psdct(x1,x2)


@given(vec_pair=make_num_den(tri=True))
def test_pseudoct_minconnect_(vec_pair):
    x1,x2 = vec_pair
    psdct = pseudocount("min-connect")
    psdct(x1,x2)


@given(vec_pair=make_num_den(),a=st.floats(min_value=0., max_value=1.))
def test_pseudoct_zerosum_(vec_pair,a):
    x1,x2 = vec_pair
    psdct = pseudocount(('zero-sum',a))
    psdct(x1,x2)
