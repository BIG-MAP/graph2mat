from collections.abc import Mapping

from e3nn import o3

from ._stored_rtps import ALL_RTPS


class LazyDict(Mapping):
    """Wrapper around the rtps dictionary that computes the rtps if they are not stored.

    Maybe this could become a memory bottleneck (??) because we store all rtps.
    Let's see.
    """

    def __init__(self, raw_dict):
        self._raw_dict = raw_dict

    def __getitem__(self, key: tuple[str, str, str]):
        if key not in self._raw_dict:
            i_irrep, j_irrep, symmetry = key
            rtp = o3.ReducedTensorProducts(symmetry, i=i_irrep, j=j_irrep)

            self._raw_dict[(i_irrep, j_irrep, symmetry)] = {
                "change_of_basis": rtp.change_of_basis,
                "irreps_out": rtp.irreps_out,
            }

        return self._raw_dict[key]

    def __setitem__(self, key, value):
        self._raw_dict[key] = value

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


ALL_RTPS = LazyDict(ALL_RTPS)
