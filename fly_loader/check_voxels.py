#%%%
from funlib.persistence import Array,prepare_ds
from skimage import data
import gunpowder as gp
import zarr

blobs_a = data.binary_blobs(length=300, blob_size_fraction=0.05, n_dim=2)


raw_a_s1 = Array(blobs_a[::2, ::2], offset=(0, 0), voxel_size=(2, 2))
# %%
blobs_a.shape
# %%
ds = prepare_ds("/groups/cellmap/cellmap/zouinkhim/GuidedNet/fly_loader/tmp.zarr/raw", shape = blobs_a.shape, offset=gp.Coordinate(0,0),voxel_size=gp.Coordinate(2,2),dtype=blobs_a.dtype,chunk_shape=[128,128],mode="w")

# %%
ds[:] = blobs_a
# %%
from gunpowder import Batch, BatchRequest, Array, ArrayKey
class ResetResolution(gp.BatchFilter):

    def __init__(self, source):
        assert isinstance(source, gp.ArrayKey)
        self.source = source

    def prepare(self, request):
        source_request = self.spec[self.source].copy()
        source_request.voxel_size = gp.Coordinate((1,) * source_request.roi.dims)
        deps = BatchRequest()
        deps[self.source] = source_request

        return deps

    def process(self, batch, request):
        batch.arrays[self.source].spec.voxel_size = gp.Coordinate((1,) * batch.arrays[self.source].spec.roi.dims)
        return batch
    
import gunpowder as gp

raw = gp.ArrayKey('RAW')

source = gp.ZarrSource(
    '/groups/cellmap/cellmap/zouinkhim/GuidedNet/fly_loader/tmp.zarr',  # the zarr container
    {raw: 'raw'},  # which dataset to associate to the array key
    {raw: gp.ArraySpec(interpolatable=True)}  # meta-information
)
random_location = gp.RandomLocation()

pipeline = source +  ResetResolution(source=raw)
# + random_location

request = gp.BatchRequest()
input_size = gp.Coordinate(64, 128)
request.add(raw, input_size, voxel_size=gp.Coordinate(2,2))

with gp.build(pipeline):
  batch = pipeline.request_batch(request)

import matplotlib.pyplot as plt
plt.imshow(batch[raw].data)
batch
# %%