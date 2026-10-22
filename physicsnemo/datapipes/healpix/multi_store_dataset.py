"""Concatenation of per-store HEALPix datasets built with the same split dates.

Training on several stores (for example two climate runs with the same grid and
variables) concatenates one dataset per store. The datasets batch internally,
so each item is a batch from a single store and a shuffled sampler mixes the
stores between steps.

Callers outside the dataset (the rollout validator, constants lookup) read
per-store attributes such as ``time_da``, ``start_index`` or ``ds``. Those reads
resolve to the first store, so item indices computed from them address that
store's part of the concatenation. The rollout bookkeeping attributes the
validator writes go to the first store for the same reason.
"""

from torch.utils.data import ConcatDataset


class MultiStoreDataset(ConcatDataset):
    """``ConcatDataset`` of same-layout datasets that reads attributes from the first."""

    # Rollout state that callers set on the dataset between items.
    _PRIMARY_STATE = ("curr_item", "integration_step", "last_batch")

    def __init__(self, datasets):
        datasets = list(datasets)
        if len(datasets) < 2:
            raise ValueError("MultiStoreDataset needs at least two datasets")
        first = datasets[0]
        for attr in ("batch_size", "spatial_dims", "input_variables", "output_variables"):
            ref = getattr(first, attr, None)
            for i, ds in enumerate(datasets[1:], start=1):
                if getattr(ds, attr, None) != ref:
                    raise ValueError(
                        f"store {i} differs from store 0 in {attr!r}: "
                        f"{getattr(ds, attr, None)!r} != {ref!r}"
                    )
        super().__init__(datasets)

    @property
    def primary(self):
        return self.datasets[0]

    def __getattr__(self, name):
        # Only reached when normal lookup fails. Dunders and the ConcatDataset
        # fields must not recurse (unpickling in workers runs before they exist).
        if name.startswith("__") or name in ("datasets", "cumulative_sizes"):
            raise AttributeError(name)
        return getattr(self.datasets[0], name)

    def __setattr__(self, name, value):
        if name in self._PRIMARY_STATE:
            setattr(self.datasets[0], name, value)
        else:
            super().__setattr__(name, value)
