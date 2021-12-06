import pytest
from pathlib import Path
from vacation_router.gen_algo import perturb_bitmask, get_valid_puturbed_masks, get_best_perturbed_masks

def test_perturb_bitmask():
    res = perturb_bitmask(0b1111, lam=5, belong_to=set(range(16)))
    assert res < 16
    # if I disallow all the bits, I should get a 0
    res = perturb_bitmask(0b1111, lam=5, belong_to=set(range(16)), exclude_mask=0b111111111)
    assert res == 0
 
def test_get_valid_perturbed_masks():
    test_masks = [0b10000101000, 0b01010101010, 0b10101010101]
    new_masks = get_valid_puturbed_masks(test_masks, lam=5, valid_bitmasks=set(range(1000000)))
    assert len(new_masks) == 3
    assert new_masks[0] & new_masks[1] & new_masks[2] == 0

def test_get_best_perturbed_masks():
    score_func = lambda x: x
    test_masks = [0b10000101000, 0b01010101010, 0b10101010101, 0b11111111111]
    new_masks = get_best_perturbed_masks(test_masks, lam=5, valid_bitmasks=range(1000000), score_func=score_func)
    assert len(new_masks) == 4
    # check that there is no bit clashing
    assert new_masks[0] & (new_masks[1] | new_masks[2] | new_masks[3]) == 0
    assert new_masks[1] & (new_masks[0] | new_masks[2] | new_masks[3]) == 0
    assert new_masks[2] & (new_masks[0] | new_masks[1] | new_masks[3]) == 0
    assert new_masks[3] & (new_masks[0] | new_masks[1] | new_masks[2]) == 0

    # check that the score of the last mask is the highest
    assert score_func(new_masks[3]) > max(score_func(new_masks[0]), score_func(new_masks[1]), score_func(new_masks[2]))
