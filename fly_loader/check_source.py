#%%
import logging
from pathlib import Path

import fibsem_tools as fst
import gunpowder as gp
import numpy as np
import xarray as xr
from scipy.ndimage import zoom
from fly_organelles.guided_data import spatial_spec_from_xarray
from fly_organelles.utils import (
    corner_offset,
    find_target_scale,
    find_target_scale_by_offset,
    get_scale_info,
)

logger = logging.getLogger(__name__)
    

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
        # self.specs[low_labelkey] =  gp.array_spec.ArraySpec(
        #     roi=label_roi, voxel_size=label_voxel_size, interpolatable=False, dtype=self.stores[low_labelkey].dtype
        # )
        self.specs[low_labelkey] =  gp.array_spec.ArraySpec(
            roi=label_roi, voxel_size=self.low_voxel_size, interpolatable=False, dtype=self.stores[low_labelkey].dtype
        )
        self.low_res_specs = gp.array_spec.ArraySpec(
            roi=label_roi, voxel_size=self.low_voxel_size, interpolatable=False, dtype=self.stores[low_labelkey].dtype
        )
        self.upsample_factor = low_sampling[0] / high_sampling[0]
        self.raw_arraykey = raw_arraykey

        self.padding += gp.Coordinate(max(0, p) for p in self.max_request - (cropsize + self.padding * 2)) / 2.0
        self.specs[raw_arraykey] = raw_spec
        for k, v in self.specs.items():
            

            v.roi = v.roi/gp.Coordinate(v.voxel_size)
            v.voxel_size = gp.Coordinate((1,)*len(v.voxel_size))


        self.low_labelkey = low_labelkey
        self.low_sampling = gp.Coordinate(low_sampling)

    def get_size(self):
        return self.stores[self.high_labelkey].size

    def setup(self):
        for key, spec in self.specs.items():
            # if key == self.low_labelkey:
            #     spec.roi = (spec.roi / self.low_sampling) * self.low_sampling
            print(f"Providing {key} with {spec}")
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
                # vs = self.low_res_specs.voxel_size
                dataset_roi = rs.roi
                dataset_roi = (dataset_roi / vs) / gp.Coordinate((self.upsample_factor,)*len(vs))
                dataset_roi = dataset_roi - self.low_res_specs.roi.offset / vs
                arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])
                arr = zoom(arr, self.factor, mode='nearest', order=0)
            else:
                dataset_roi = dataset_roi / vs
                dataset_roi = dataset_roi - self.specs[ak].roi.offset / vs
                logger.debug(f"Reading {ak} with dataset_roi {dataset_roi} ({dataset_roi.to_slices()})")
                # loc = {axis:slice(b, e, None) for b, e, axis in zip(rs.roi.get_begin(), rs.roi.get_end()-vs/2., "zyx")}
                # arr = self.stores[ak].sel(loc).to_numpy()
            
                arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])

            logger.debug(f"Read array of shape {arr.shape}")
            array_spec = self.specs[ak].copy()
            array_spec.roi = rs.roi
            # array_spec.voxel_size = gp.Coordinate((1,) * rs.roi.dims)
            batch.arrays[ak] = gp.Array(arr, array_spec)
        timing.stop()
        batch.profiling_stats.add(timing)
        return batch
    

#%%
import neuroglancer
import numpy as np

def view_in_neuroglancer(**kwargs):
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for array_name, array in kwargs.items():
            if (
                array.dtype in (float, np.float32)
                or "raw" in array_name
                or "__img" in array_name
            ):
                s.layers[array_name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(
                        data=array,
                    ),
                )
            else:
                s.layers[array_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=array,
                    ),
                )
    print(viewer.get_viewer_url())
#%%
raw_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/"
gt_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations"
label = "mito"
high_sampling = (16, 16, 16)
low_sampling = (32, 32, 32)
#%%
import gunpowder as gp
import numpy as np


raw_key = gp.ArrayKey("RAW")
high_label_key = gp.ArrayKey("LABEL_HIGH")
low_label_key = gp.ArrayKey("LABEL_LOW")


def make_data_pipeline(
    label: str,
    raw_path: str,
    gt_path: str,
    pad_width_out: gp.Coordinate,
    low_sampling: tuple[int],
    high_sampling: tuple[int],
    max_out_request: gp.Coordinate,
):

    src = GuidedCellMapCropSource(
        gt_path,
        raw_path,
        label,
        raw_key,
        low_label_key,
        high_label_key,
        high_sampling,
        low_sampling,
        base_padding=pad_width_out,
        max_request=max_out_request,
    )
    src_pipe = src
        
    # src_pipe+= AverageUpSample(
    #     source=low_label_key,
    #     target_voxel_size=gp.Coordinate(high_sampling),
    # )
    

    src_pipe += gp.RandomLocation()

    return src_pipe

#%%
input_size = gp.Coordinate((100, 100, 100))
output_size = gp.Coordinate((100, 100, 100))
# * gp.Coordinate((2, 2, 2))
input_size = gp.Coordinate(input_size)
# * gp.Coordinate(high_sampling)
output_size = gp.Coordinate(output_size) 
# * gp.Coordinate(low_sampling)
pad_width_out = output_size / 2.0
displacement_sigma = gp.Coordinate((24, 24, 24))
# max_in_request = gp.Coordinate((np.ceil(np.sqrt(sum(input_size**2))),)*len(input_size)) + displacement_sigma * 6
max_out_request = (
    gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size)) + displacement_sigma * 6
)
pipeline = make_data_pipeline(
    label=label,
    raw_path=raw_path,
    gt_path=gt_path,
    pad_width_out=pad_width_out,
    low_sampling=low_sampling,
    high_sampling=high_sampling,
    max_out_request=max_out_request,
)
#%%

request = gp.BatchRequest()

request.add(low_label_key, output_size, voxel_size=gp.Coordinate(1,1,1))
request.add(raw_key, input_size, voxel_size=gp.Coordinate(1,1,1))
request.add(high_label_key, input_size, voxel_size=gp.Coordinate(1,1,1))
#%%
result = []
with gp.build(pipeline) as pp:
    while True:
        print("Requesting batch...")
        batch = pp.request_batch(request)

        if batch[high_label_key].data.any():
            print("Batch received with high label data.")
            result.append(batch)
            if len(result) >= 2:
                break
# %%
batch[high_label_key].data.any()


# %%
batch = result[1]
view_in_neuroglancer(
    raw=batch[raw_key].data,
    label_high=batch[high_label_key].data,
    label_low=batch[low_label_key].data,
    # expended_high= new_array,
)
# %%
batch[high_label_key].data
#%%
array = batch[high_label_key].data
from scipy.ndimage import zoom

# %%
array.shape
# %%
new_array.shape
# %%
