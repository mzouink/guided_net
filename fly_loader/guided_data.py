import logging
from pathlib import Path

import fibsem_tools as fst
import gunpowder as gp
import numpy as np
import xarray as xr
from skimage.transform import pyramid_expand

from fly_organelles.utils import (
    corner_offset,
    find_target_scale,
    find_target_scale_by_offset,
    get_scale_info,
)

logger = logging.getLogger(__name__)


def spatial_spec_from_xarray(xarr) -> tuple[gp.Roi, gp.Coordinate]:
    if not isinstance(xarr, xr.DataArray):
        msg = f"Expected input to be `xarray.DataArray`, not {type(xarr)}"
        raise TypeError(msg)
    offset = []
    for axis in "zyx":
        offset.append(int(xarr.coords[axis][0].data))
    offset = gp.Coordinate(offset)
    voxel_size = []
    for axis in "zyx":
        voxel_size.append(int((xarr.coords[axis][1] - xarr.coords[axis][0]).data))
    voxel_size = gp.Coordinate(voxel_size)
    shape = voxel_size * gp.Coordinate(xarr.shape)
    roi = gp.Roi(offset, shape)
    return roi, voxel_size


class GuidedCellMapCropSource(gp.batch_provider.BatchProvider):
    def __init__(
        self,
        label_store: str,
        raw_store: str,
        label: str,
        raw_arraykey: gp.ArrayKey,
        low_labelkey: gp.ArrayKey,
        high_labelkey: gp.ArrayKey,
        high_sampling: tuple[int],
        low_sampling: tuple[int],
        base_padding: gp.Coordinate,
        max_request: gp.Coordinate,
    ):
        super().__init__()
        self.stores = {}
        self.specs = {}
        self.max_request = max_request
        raw_grp = fst.read(raw_store)
        raw_scale, raw_offset, raw_shape = find_target_scale(raw_grp, high_sampling)
        raw_offset = gp.Coordinate((0,) * len(high_sampling))  # tuple(np.array(raw_offset) - np.array(high_sampling)/2.)
        raw_native_scale = get_scale_info(raw_grp)[1]["s0"]
        self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale) 
        raw_roi = gp.Roi(raw_offset, gp.Coordinate(high_sampling) * gp.Coordinate(raw_shape))
        raw_voxel_size = gp.Coordinate(high_sampling)
        # raw_roi, raw_voxel_size = spatial_spec_from_xarray(self.stores[raw_arraykey])
        raw_spec = gp.array_spec.ArraySpec(
            roi=raw_roi, voxel_size=raw_voxel_size, interpolatable=True, dtype=self.stores[raw_arraykey].dtype
        )
        self.padding = base_padding
        label_grp = fst.read(Path(label_store) / label)
        label_scale, label_offset, label_shape = find_target_scale(label_grp, high_sampling)
        low_scale, low_offset, low_shape = find_target_scale(label_grp, low_sampling)
        label_offset = gp.Coordinate(
            corner_offset(np.array(label_offset), np.array(raw_native_scale), np.array(high_sampling))
        )
        if label_offset % raw_voxel_size == gp.Coordinate((0,) * len(high_sampling)):
            self.needs_downhigh_sampling = False
            self.secret_raw_offset = gp.Coordinate((0,) * len(high_sampling))
        else:
            self.needs_downhigh_sampling = True
            logger.debug(f"Need to downsample raw for {label_store} to accomodate offset {label_offset}.")
            raw_scale, raw_offset, raw_res, raw_shape = find_target_scale_by_offset(raw_grp, label_offset)
            logger.debug(f"Reading raw from {raw_store}/ {raw_scale} with voxel_size {raw_res}")
            self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale)
            raw_roi = gp.Roi(
                gp.Coordinate((0,) * len(high_sampling)),
                gp.Coordinate(raw_shape) * gp.Coordinate(raw_res) - gp.Coordinate(high_sampling),
            )
            raw_spec = gp.array_spec.ArraySpec(
                roi=raw_roi, voxel_size=raw_res, interpolatable=True, dtype=self.stores[raw_arraykey].dtype
            )
            self.secret_raw_offset = label_offset % gp.Coordinate(high_sampling)
            label_offset -= self.secret_raw_offset
        self.high_labelkey = high_labelkey
        # label_offset = tuple(np.array(label_offset) - np.array(high_sampling)/2.)
        self.stores[high_labelkey] = fst.read_xarray(Path(label_store) / label / label_scale)
        self.stores[low_labelkey] = fst.read_xarray(Path(label_store) / label / low_scale)
        # label_roi, label_voxel_size = spatial_spec_from_xarray(self.stores[high_labelkey])
        cropsize = gp.Coordinate(label_shape) * gp.Coordinate(high_sampling)
        label_roi = gp.Roi(label_offset, cropsize)

        label_voxel_size = gp.Coordinate(high_sampling)
        self.low_voxel_size = gp.Coordinate(low_sampling)
        self.high_sampling = gp.Coordinate(high_sampling)
        self.factor =self.low_voxel_size[0]/ self.high_sampling[0]
        self.specs[high_labelkey] = gp.array_spec.ArraySpec(
            roi=label_roi, voxel_size=label_voxel_size, interpolatable=False, dtype=self.stores[high_labelkey].dtype
        )
        self.specs[low_labelkey] =  gp.array_spec.ArraySpec(
            roi=label_roi, voxel_size=label_voxel_size, interpolatable=False, dtype=self.stores[low_labelkey].dtype
        )
        self.low_res_specs = gp.array_spec.ArraySpec(
            roi=label_roi, voxel_size=self.low_voxel_size, interpolatable=False, dtype=self.stores[low_labelkey].dtype
        )
        self.upsample_factor = low_sampling[0] / high_sampling[0]
        self.raw_arraykey = raw_arraykey

        self.padding += gp.Coordinate(max(0, p) for p in self.max_request - (cropsize + self.padding * 2)) / 2.0
        self.specs[raw_arraykey] = raw_spec
        self.low_labelkey = low_labelkey

    def get_size(self):
        return self.stores[self.high_labelkey].size

    def setup(self):
        for key, spec in self.specs.items():
            self.provides(key, spec)

    def provide(self, request) -> gp.Batch:
        timing = gp.profiling.Timing(self)
        timing.start()
        batch = gp.batch.Batch()
        for ak, rs in request.array_specs.items():
            logger.debug(f"Requesting {ak} with {rs}")
            vs = self.specs[ak].voxel_size

            if ak == self.raw_arraykey:
                dataset_roi = rs.roi + self.secret_raw_offset
                logger.debug(f"Shifting {ak} dataset_roi by secret raw offset {dataset_roi}")
            else:
                dataset_roi = rs.roi
            if ak == self.low_labelkey:
                vs = self.low_res_specs.voxel_size
                dataset_roi = rs.roi
                dataset_roi = dataset_roi / vs
                dataset_roi = dataset_roi - self.low_res_specs.roi.offset / vs
                arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])
                arr = pyramid_expand(arr, self.factor).astype(arr.dtype)
            else:
                dataset_roi = dataset_roi / vs
                dataset_roi = dataset_roi - self.spec[ak].roi.offset / vs
                logger.debug(f"Reading {ak} with dataset_roi {dataset_roi} ({dataset_roi.to_slices()})")
                # loc = {axis:slice(b, e, None) for b, e, axis in zip(rs.roi.get_begin(), rs.roi.get_end()-vs/2., "zyx")}
                # arr = self.stores[ak].sel(loc).to_numpy()
            
                arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])

            logger.debug(f"Read array of shape {arr.shape}")
            array_spec = self.specs[ak].copy()
            array_spec.roi = rs.roi
            batch.arrays[ak] = gp.Array(arr, array_spec)
        timing.stop()
        batch.profiling_stats.add(timing)
        return batch
    


import logging

import gunpowder as gp


logger = logging.getLogger(__name__)


class AverageUpSample(gp.BatchFilter):
    """Upsample arrays in a batch by given factors.

    Args:

        source (:class:`ArrayKey`):

            The key of the array to downsample.

        factor (``int`` or ``tuple`` of ``int``):

            The factor to downsample with.

        target (:class:`ArrayKey`):

            The key of the array to store the downsampled ``source``.
    """

    def __init__(self, source, target_voxel_size, target=None):
        assert isinstance(source, gp.ArrayKey)
        self.source = source
        self.target_voxel_size = gp.Coordinate(target_voxel_size)
        if target is None:
            self.target = source
        else:
            assert isinstance(target, gp.ArrayKey)
            self.target = target

    def setup(self):

        self.source_voxel_size = self.get_upstream_provider().spec.array_specs[self.source].voxel_size
        # self.factor = self.source_voxel_size / self.target_voxel_size 
        self.factor = tuple(a / b for a, b in zip(self.source_voxel_size, self.target_voxel_size))

        spec = self.spec[self.source].copy()
        spec.voxel_size = self.target_voxel_size
        source_roi = spec.roi
        spec.roi = (spec.roi / self.source_voxel_size) * self.source_voxel_size
        logger.debug(f"Updating {source_roi} to {spec.roi}")
        # assert self.target_voxel_size % self.source_voxel_size == gp.Coordinate(
        #     (0,) * len(self.target_voxel_size)
        # ), f"{self.target_voxel_size % self.source_voxel_size}"
        assert self.target_voxel_size < self.source_voxel_size
        # if not spec.interpolatable:
        #     msg = "can't use average upsampling for non-interpolatable arrays"
        #     raise ValueError(msg)

        if self.target == self.source:
            self.updates(self.target, spec)
        else:
            self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):
        # intialize source request with existing request for target
        source_request = request[self.target].copy()
        # correct the voxel size for source
        logger.debug(f"Initializing source request with {source_request}")
        # source_voxel_size = self.spec[self.source].voxel_size
        source_request.voxel_size = self.source_voxel_size
        deps = gp.BatchRequest()
        deps[self.source] = source_request
        return deps

    def process(self, batch, request):
        source = batch.arrays[self.source]
        data = source.data
        src_dtype = data.dtype

        channel_dims = len(data.shape) - source.spec.roi.dims
        factor = (1,) * channel_dims + self.factor
        resampled_data = pyramid_expand(data, factor).astype(src_dtype)
        logger.debug(f"Downsampling turns shape {data.shape} into {resampled_data.shape}")
        target_spec = source.spec.copy()
        target_spec.roi = gp.Roi(
            source.spec.roi.get_begin(),
            self.target_voxel_size * gp.Coordinate(resampled_data.shape[-self.target_voxel_size.dims :]),
        )
        target_spec.voxel_size = self.target_voxel_size
        logger.debug(f"returning array with spec {target_spec}")

        # create output array
        outputs = gp.Batch()
        outputs.arrays[self.target] = gp.Array(resampled_data, target_spec)

        return outputs
