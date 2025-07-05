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
high_sampling = (8, 8, 8)
low_sampling = (16, 16, 16)
#%%
import gunpowder as gp
import numpy as np

from fly_organelles.guided_data import GuidedCellMapCropSource, AverageUpSample



def make_data_pipeline(
    label: str,
    raw_path: str,
    gt_path: str,
    pad_width_out: gp.Coordinate,
    low_sampling: tuple[int],
    high_sampling: tuple[int],
    max_out_request: gp.Coordinate,
):
    raw_key = gp.ArrayKey("RAW")
    high_label_key = gp.ArrayKey("LABEL_HIGH")
    low_label_key = gp.ArrayKey("LABEL_LOW")
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
input_size = gp.Coordinate((178, 178, 178))
output_size = gp.Coordinate((56, 56, 56))
input_size = gp.Coordinate(input_size) * gp.Coordinate(high_sampling)
output_size = gp.Coordinate(output_size) * gp.Coordinate(high_sampling)
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
raw_key = gp.ArrayKey("RAW")
high_label_key = gp.ArrayKey("LABEL_HIGH")
low_label_key = gp.ArrayKey("LABEL_LOW")
request = gp.BatchRequest()
request.add(raw_key, input_size, voxel_size=gp.Coordinate(high_sampling))
request.add(high_label_key, output_size, voxel_size=gp.Coordinate(high_sampling))
request.add(low_label_key, output_size, voxel_size=gp.Coordinate(high_sampling))
#%%
result = []
with gp.build(pipeline) as pp:
    while True:
        print("Requesting batch...")
        batch = pp.request_batch(request)
        if batch[high_label_key].data.any():
            print("Batch received with high label data.")
            result.append(batch)
            if len(result) >= 10:
                break
# %%
batch[high_label_key].data.any()


# %%
batch = result[0]
view_in_neuroglancer(
    raw=batch[raw_key].data,
    label_high=batch[high_label_key].data,
    label_low=batch[low_label_key].data,
)
# %%
