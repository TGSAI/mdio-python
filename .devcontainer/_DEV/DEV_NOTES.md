Dmitriy Repin - EPAM

## Comment 1
Here are my thoughts while looking at the sketch:
 
1. We should have a single point of entry for standard and custom schemas
2. blocked_io.to_zarr will need some heavy modification
  We need to make sure coordinates are written in parallel with the primary Variable.  
  This doesn't write the dimension arrays. I think we need to use the actual Zarr API to handle this, Xarray enforces immutability on these arrays.
3. Name assumptions
  variable_name: We assume this is the highest dimensionality data
  header_array: We assume only one "structured" data type
4. to_zarr is getting to be very overloaded. My preference would be to rename this convenience function to_mdio
  It might make sense to check for kwarg compute before blindly setting to false
 
# Comment 1
```
grid_chunksize = None  # Q: Where do we get an initial value. Do we need to expose them from mdio_template?
```
Looks like this is only used in grid overrides (and rarely)? We may need to re-think how this is done. 
 
```
- Validate the specified MDIO template matches the SegySpec (how?)
```
I think we just need to ensure the dimension names and coordinate names are in the SegySpec. That should be enough I believe?
 
 

    # Write traces to the MDIO Zarr dataset
    # Currently the name of the variable in the dataset for the data volume is:
    # e.g. "StackedAmplitude" for 2D post-stack depth
    # e.g. "StackedAmplitude" for 3D post-stack depth
    # e.g. "AmplitudeCDP" for 3D pre-stack CPD depth
    # e.g. "AmplitudeShot" for 3D pre-stack time Shot gathers
    variable_name = "Amplitude" #TODO: Use the proper variable name
variable name should be defined in the AbstractSeismicTemplate. I am ok with setting it to amplitude by default. if users want to customize it, that's fine.
 
header_array=None, #TODO: where do we get the header array from?
the header array will be created by mdio factory. it is just the handle to the array.
 
# {"mean": glob_mean, "std": glob_std, "rms": glob_rms, "min": glob_min, "max": glob_max} these are the old v0 stats. v1 has different ones
 
class DynamicallyLoadedModule can you explain this? are we loading arbitrary .py files?
 
 

def segy_to_mdio_v1_custom(
    input: StorageLocation,
    output: StorageLocation,
    segy_spec: str | StorageLocation,
    mdio_template: str | StorageLocation,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None, 
    index_types: Sequence[str] | None = None,
    overwrite: bool = False,
):
 
index_* parameters in the function signature should be gone, these are all encapsulated in segy spec and mdio template. it seems like this is a convenience function to override segy spec for non-savvy users? I think we can do it more general with CLI, i don't see the need for this at the moment
 
Brian Michell
Here are my thoughts while looking at the sketch          1. We should have a single point of entry for standard and custom schemas2. blocked_io.to_zarr will need some heavy modification  We need to make sure coordinates are written in parallel with the primary Variable.    This doesn't write the d…
regarding Brian's comments;
	1	yes
	2	coordinates have different chunking and can't write in parallel efficiently (an no need to). they can be scanned during initial scan (which is parallel reads) and be written in full before trace workers start. They're small. that is in the prototype implementation I provided
	3	I mentioned above it should be in the template and customizable. Header array name should also be in the template.
	4	Agreed, let's call it to_mdio or to_disk etc, but small detail
 
 
if we want to be more generic; we can call the variables trace_data and trace_headers, but IMO its too generic, would be nice to customize. I like seismic or amplitude as default though.
 
wdyt
 
 

ds = xr.Dataset(
    {
        'amplitude': (['inline', 'crossline', 'time'], data),
        'velocity': (['inline', 'crossline', 'time'], velocity)
        'salt_mask': (['inline', 'crossline', 'time'], segmentation_mask,
        'headers': (['inline', 'crossline'], header_data,
    },
    coords={
        'inline': ilines,
        'crossline': xlines,
        'time': times,
        'cdp_x': (['inline', 'crossline'], x_coord),
        'cdp_y': (['inline', 'crossline'], y_coord),
    }
)
 
we could do nice things like this
 

Dmitriy Repin - EPAM
Could you please take a look at a rough sketch.  Maybe we could discuss this tomorrow at the stand up    src/mdio/converters/segy_to_mdio_v1.py  def segy_to_mdio_v1(      input: StorageLocation,      output: StorageLocation,      segy_spec: SegySpec,      mdio_template: AbstractDatasetTemplate,    …
Here are my thoughts while looking at the sketch
 

1. We should have a single point of entry for standard and custom schemas
2. blocked_io.to_zarr will need some heavy modification
  We need to make sure coordinates are written in parallel with the primary Variable.  
  This doesn't write the dimension arrays. I think we need to use the actual Zarr API to handle this, Xarray enforces immutability on these arrays.
3. Name assumptions
  variable_name: We assume this is the highest dimensionality data
  header_array: We assume only one "structured" data type
4. to_zarr is getting to be very overloaded. My preference would be to rename this convenience function to_mdio
  It might make sense to check for kwarg compute before blindly setting to false
 
 
grid_chunksize = None  # Q: Where do we get an initial value. Do we need to expose them from mdio_template?
Looks like this is only used in grid overrides (and rarely)? We may need to re-think how this is done. 
 
# Validate the specified MDIO template matches the SegySpec (how?)
I think we just need to ensure the dimension names and coordinate names are in the SegySpec. That should be enough I believe?
 
 

    # Write traces to the MDIO Zarr dataset
    # Currently the name of the variable in the dataset for the data volume is:
    # e.g. "StackedAmplitude" for 2D post-stack depth
    # e.g. "StackedAmplitude" for 3D post-stack depth
    # e.g. "AmplitudeCDP" for 3D pre-stack CPD depth
    # e.g. "AmplitudeShot" for 3D pre-stack time Shot gathers
    variable_name = "Amplitude" #TODO: Use the proper variable name
variable name should be defined in the AbstractSeismicTemplate. I am ok with setting it to amplitude by default. if users want to customize it, that's fine.
 
header_array=None, #TODO: where do we get the header array from?
the header array will be created by mdio factory. it is just the handle to the array.
 
# {"mean": glob_mean, "std": glob_std, "rms": glob_rms, "min": glob_min, "max": glob_max} these are the old v0 stats. v1 has different ones
 
class DynamicallyLoadedModule can you explain this? are we loading arbitrary .py files?
 
 

def segy_to_mdio_v1_custom(
    input: StorageLocation,
    output: StorageLocation,
    segy_spec: str | StorageLocation,
    mdio_template: str | StorageLocation,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None, 
    index_types: Sequence[str] | None = None,
    overwrite: bool = False,
):
 
index_* parameters in the function signature should be gone, these are all encapsulated in segy spec and mdio template. it seems like this is a convenience function to override segy spec for non-savvy users? I think we can do it more general with CLI, i don't see the need for this at the moment
 
Brian Michell
Here are my thoughts while looking at the sketch          1. We should have a single point of entry for standard and custom schemas2. blocked_io.to_zarr will need some heavy modification  We need to make sure coordinates are written in parallel with the primary Variable.    This doesn't write the d…
regarding Brian's comments;
	1	yes
	2	coordinates have different chunking and can't write in parallel efficiently (an no need to). they can be scanned during initial scan (which is parallel reads) and be written in full before trace workers start. They're small. that is in the prototype implementation I provided
	3	I mentioned above it should be in the template and customizable. Header array name should also be in the template.
	4	Agreed, let's call it to_mdio or to_disk etc, but small detail
 
 
if we want to be more generic; we can call the variables trace_data and trace_headers, but IMO its too generic, would be nice to customize. I like seismic or amplitude as default though.
 
wdyt
 
 

ds = xr.Dataset(
    {
        'amplitude': (['inline', 'crossline', 'time'], data),
        'velocity': (['inline', 'crossline', 'time'], velocity)
        'salt_mask': (['inline', 'crossline', 'time'], segmentation_mask,
        'headers': (['inline', 'crossline'], header_data,
    },
    coords={
        'inline': ilines,
        'crossline': xlines,
        'time': times,
        'cdp_x': (['inline', 'crossline'], x_coord),
        'cdp_y': (['inline', 'crossline'], y_coord),
    }
)
 
we could do nice things like this
 