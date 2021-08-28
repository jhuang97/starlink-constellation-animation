# starlink-constellation-animation
Animated scatter plots of Starlink satellite orbits

I used this code to make this video: https://youtu.be/zYPJtTgS060

This all started out as an attempt to recreate Elias Eccli's Starlink animations (http://youtu.be/rddTXl_7Wr8).

Procedure
1. Make an account on space-track.org and download the SGP4 propagator. From the SGP4 propagator that you downloaded, copy `SampleCode/Python/DriverExamples/Sgp4Prop/src/wrappers/`, you need to copy into the `src` folder of this project. (`AstroUtils.py` also comes from there)
2. `shell_commands.txt` -- download satellite info and TLEs from space-track.org
3. `get_starlink_launch_info.py` -- Get list of Starlink launches from, err, Wikipedia
4. `get_starlink_ids.py` -- Make list of NORAD IDs of Starlink launches
5. `filter_save_tles.py` -- Parse the TLEs, exclude the ones that look bad, and save the remaining TLE data as numpy arrays in .npz files
6. `load_filtered_tles.py` -- Load some filtered TLE data, propagate using SGP4 propagator, make some plots  
 6.1. This script requires setting the environment variable `LD_LIBRARY_PATH` to the path to your SGP4 propagator.  
 6.2. At this point, you might want to clean the data more. You might also want to adjust `..\input\reference_satellite_1.inp` to your liking (I came up with this by much trial and error).
7. `make_animation_data.py` -- Make and save the information needed for the animated scatter plots.  
 7.1. Also requires setting the environment variable `LD_LIBRARY_PATH` to the path to your SGP4 propagator.
9. `export_animation.py` -- Load the animation data, make the animations, save as .mp4 videos.  This can take hours to run.

## License
The code that comes from the SGP4 propagator (`src/AstroUtils.py` and parts of `src/SatProp.py`) are under the SGP4 Open License.  All other code in this repo is released under the license in the following file: [LICENSE](LICENSE).
