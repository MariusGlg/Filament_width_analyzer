
# Filament_width_analyzer

The analysis can be used to determine the average width (full-width half-maximum) of an object from an image, e.g. straight filamenteous (microtubuli or vimentin). 

- Draws a line along a filament
- generates equally spaced lines perpendicular to the line along the filament
- fits a Gaussian function along all intensity profiles defined by perpendicular lines as a filtering step
- Averages all itensity distributions and performs a Gaussian fit to extract the FWHM of the averaged profile

Requirements: numpy, skimage, matplotlib, scipy, numpy
Input file: Image (.tiff)
Execution: Filament_width_analyer.py config.ini

Config file: 
[INPUT_FILES]
path - define the path to an image (e.g. example_image.tiff)
[PARAMETERS]
px_size - image pixel size in µm
start_x_coordinate - x-coordinate defining the startpoint of the line along the filament
start_y_coordinate - y-coordinate defining the startpoint of the line along the filament
stop_x_coordinate - x-coordinate defining the endpoint of the line along the filament
stop_y_coordinate - y-coordinate defining the endpoint of the line along the filament
len_perp_line - length of lines perpendicular to the line along the filament
line_segments - number of perpendicular lines


