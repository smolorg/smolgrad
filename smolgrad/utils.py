def broadcast_axis(left, right):
    """
    mlx uses broadcasting before performing array ops
    this function determines which axes on either arrays will be broadcasted
    in order to calculate gradients along those axes.

    example:
    >>> left.shape = (3, 1)
    >>> right.shape = (1, 4)
    >>> broadcast_axis(left, right)     # ((1, ), (0, ))

    here the second axis for left, and first axis for right will be broadcasted
    """
    
    ldim = len(left)
    rdim = len(right)
    maxdim = max(ldim, rdim)

    lshape_new = (1, ) * (maxdim - ldim) + left
    rshape_new = (1, ) * (maxdim - rdim) + right

    assert len(lshape_new) == len(rshape_new)

    left_axes, right_axes = [], []

    for i in range(len(lshape_new)):
        if lshape_new[i] > rshape_new[i]:
            right_axes.append(i)
        elif rshape_new[i] > lshape_new[i]:
            left_axes.append(i)

    return tuple(left_axes), tuple(right_axes)
