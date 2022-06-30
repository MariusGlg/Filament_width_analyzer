"""
@author: Marius Glogger
Research Group Heilemann
Institute for Physical and Theoretical Chemistry, Goethe University Frankfurt a.M.

Determines the with (FWHM) of filaments in a fluorescence image and displays the results.
"""

from configparser import ConfigParser
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings

#  config.ini file
file = "config.ini"
config = ConfigParser()
config.read(file)

#  config file parameter
#  [x,y] coordinates of start and endpoints (start and stop) of a line along a filament (vimentin/microtubuli etc.)
#  len_perp_line = length of perpendicular lines
#  line_segments = number of perpendicular lines equally spaced on the line
px_size = float(config["PARAMETERS"]["px_size"])
start = [int(config["PARAMETERS"]["start_x_coordinate"]), int(config["PARAMETERS"]["start_y_coordinate"])]
stop = [int(config["PARAMETERS"]["stop_x_coordinate"]), int(config["PARAMETERS"]["stop_y_coordinate"])]
len_perp_lines = int(config["PARAMETERS"]["len_perp_lines"])
line_segments = int(config["PARAMETERS"]["line_segments"])
# load image
img = io.imread(config["INPUT_FILES"]["path"])

#  set initial fit parameter
x = np.arange(-(len_perp_lines * px_size), (len_perp_lines * px_size), px_size)  # x data range
amp = 1  # amplitude
xc = np.median(x)  # center
sigma = x[-1] / 8  # sigma


def perpendicular_lines(start, stop, len_perp_lines, line_segments):
    """ Generates lines perpendicular to the line that follows the filamenteous structure (profile line).
    The perpendicular lines are of length defined by len_perp_line, are equally spaced
    generates lines of length len_perp_lines. The lines are perpendicular to the line defined by start/stop
    :param start: coordinates defining the start point (x,y) of the line along the filament, int
    :param stop: coordinates defining the end point (x,y) of the line along the filament, int
    :param len_perp_lines: length of the perpendicular lines in pixel, int
    :param line_segments: Number of equally spaced perpendicular lines
    :return: lists containing x,y coordinates form all perpendicular lines
    """
    linestart = []
    linestop = []
    """equally spaced subsegements of original line"""
    x_values = np.linspace(start[0], stop[0], line_segments)
    y_values = np.linspace(start[1], stop[1], line_segments)
    """coordinates of middle points of subsegments"""
    midx = (x_values[1:] + x_values[:-1]) / 2
    midy = (y_values[1:] + y_values[:-1]) / 2
    """Get a displacement vector for this segment"""
    for i in range(len(x_values)-1):
        vec = np.array([x_values[i+1] - x_values[i], y_values[i+1] - y_values[i]])
        # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)
        # Normalise the perpendicular vectors
        len_anti = ((vec_anti ** 2).sum()) ** 0.5
        vec_anti = vec_anti / len_anti
        len_clock = ((vec_clock ** 2).sum()) ** 0.5
        vec_clock = vec_clock / len_clock
        # Scale up to the profile length
        vec_anti = vec_anti * len_perp_lines
        vec_clock = vec_clock * len_perp_lines
        # append lines to array
        linestart.append([midx[i] + vec_anti[0], midy[i] + vec_anti[1]])
        linestop.append([midx[i] + vec_clock[0], midy[i] + vec_clock[1]])
    return linestart, linestop


def get_profile(linestart, linestop, image):
    """
    Extract intensity profile along perpendicular lines. The intensity profile along all perpendicular lines
    is fitted using a Gaussian function. If the fit passes certain thresholds, the intensity values are stored. From
    these values, the average intensity distribution is again fitted with a Gaussian function to determine the
    FWHM of the average profile. The average intensity distribution + fit + fluorescence image is plotted. The figure
    is saved as a .png
    :param linestart: list containing all x,y coordinates from startpoints of perpendicular lines
    :param linestop: list containing all x,y coordinates from endpoints of perpendicular lines
    :param image: fluorescence image
    :return plot-function
    """
    # Preallocate matrices
    allnorm_values = []
    all_lines_x = []
    all_lines_y = []

    for i in range(len(linestart)):
        x_line = np.linspace(linestart[i][0], linestop[i][0], len_perp_lines * 2)
        y_line = np.linspace(linestart[i][1], linestop[i][1], len_perp_lines * 2)
        w, h = image.shape  # get image Dimensions
        # check if line is not out of image boundaries
        if all(w > x > 0 for x in x_line) and all(h > y > 0 for y in y_line):
            values = image[y_line.astype(int), x_line.astype(int)]
            if not np.all(values == values[0]):  # check if signal is constant (== BG only)
                norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))  # normalize
                offset = np.min(values)
                norm_values, fit_params, cov_mat, fit_errors, fit_residual, fit_rsquared, FWHM = fit(norm_values, offset)
                # fit intensity distribution with a gaussian function. If the fit passes certain criteria
                # (user defined), intensity distribution values are stored in an array.
                # This is a prefilter step to eliminate extensive background signal, weak specific signal
                # or cross-sections
                if FWHM:  # check if output is not empty and sum data:
                    allnorm_values.append(norm_values)
                    all_lines_x.append(x_line)
                    all_lines_y.append(y_line)
    meanprofile = np.mean(allnorm_values, axis=0, dtype=np.float32)  # Average of all intensity values. We use this
    #  information for the final fit of the data.
    offset = np.mean(meanprofile)
    norm_values, fit_params, cov_mat, fit_errors, fit_residual, fit_rsquared, FWHM = fit(meanprofile, offset)
    FWHM = plot_data(meanprofile, FWHM, fit_params, all_lines_x, all_lines_y)
    return FWHM


def gaussian(x, amp, xc, sigma, offset):
    """
    gaussian function + y-offset = model used for the fit. Each intensity profile along the perpendicular lines
    is fitted with a Gaussian function.
    :param x: x-range of the fit, list
    :param amp: Amplitude, int
    :param xc: center of distribution, int
    :param sigma: sigma of gaussian function, int
    :param offset: offset of the fit, int
    :return: gaussian distribution defined by parameter
    """
    data = amp*np.exp(-(x-xc)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2) + offset
    return data


def fit(vals, offset):
    """
    fit data
    :param vals: intensity profile along perpendicular line
    :param offset: y-offset
    :return: Fit-parameter
    """
    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeWarning)
    """fit model to data"""
    init_vals = [amp, xc, sigma, offset]
    try:
        # perform the fit and calculate fit parameter errors from covariance matrix
        fit_params, cov_mat = curve_fit(gaussian, x, vals, p0=init_vals)
        fit_errors = np.sqrt(np.diag(cov_mat))
        # manually calculate R-squared goodness of fit
        fit_residual = vals - gaussian(x, *fit_params)
        fit_Rsquared = 1 - np.var(fit_residual) / np.var(vals)
        FWHM = abs(fit_params[2]*2.35)
        if fit_Rsquared > 0.50:  # filter data based on fit_rsquared and minimal FWHM values
            return vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM
        else:
            vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM = [], [], [], [], [], [], []
    except RuntimeError:
        """Return empty array"""
        vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM = [], [], [], [], [], [], []
    except RuntimeWarning:
        """Return empty array"""
        vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM = [], [], [], [], [], [], []
    except OptimizeWarning:
        """Return empty array"""
        vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM = [], [], [], [], [], [], []

    return vals, fit_params, cov_mat, fit_errors, fit_residual, fit_Rsquared, FWHM


def plot_data(meanprofile, FWHM, fit_params, all_lines_x, all_lines_y):
    """
    PLOT DATA. Average profile perpendicular to the filament is plotted together with the Gaussian fit.
    The figure is saved as .png.
    :param meanprofile: Mean intensity profile of the filament, array
    :param FWHM: FWHM of the mean intensity profile derived from the fit, int
    :param fit_params: All fit parameter
    :param all_lines_x: List containing x-coordinates from perpendicular lines
    :param all_lines_y: List containing y-coordinates from perpendicular lines
    :return FWHM
    """

    datarange = np.linspace(-(len_perp_lines * px_size), (len_perp_lines * px_size), 100)  # corrected for pixel size
    fig, axis = plt.subplots(1, 2, figsize=(9, 5))
    axis[0].imshow(img, cmap="inferno", vmax=300, aspect="auto")
    axis[0].set_title("image")
    for i in range(len(all_lines_x)):
        axis[0].plot(all_lines_x[i], all_lines_y[i], "-", markersize=2, color="white")
    plt.setp(axis[0].get_xticklabels(), visible=False)
    plt.setp(axis[0].get_yticklabels(), visible=False)
    axis[0].tick_params(axis='both', which='both', length=0)
    axis[1].bar(x, meanprofile, width=0.01, color="grey", edgecolor="black",
                alpha=0.5, label="mean profile")
    axis[1].plot(datarange, gaussian(datarange, *fit_params),
                 linewidth=2, color='black', label='fit')
    axis[1].set_title("average cross-section")
    axis[1].text(np.min(x), 1, "FWHM_fit = {} nm ".format("%.1f" % (FWHM*1000)))
    axis[1].set_ylim([0, 1.1])
    axis[1].set_xlim([np.min(datarange), np.max(datarange)])
    axis[1].set_xlabel('centered position [${\mu}m$]')
    axis[1].set_ylabel('amplitude [A.U.]')
    axis[1].legend()
    #fig.show()
    plt.savefig("vimentin_width.png")
    return FWHM


perp_linestart, perpline_stop = perpendicular_lines(start, stop, len_perp_lines, line_segments)
FWHM = get_profile(perp_linestart, perpline_stop, img)


if __name__ == "__main__":
    print("FWHM = {:.1f} nm".format(FWHM*1000))





# plt.imshow(img, cmap="inferno", vmax=300, aspect="auto")  # display settings
# plt.plot((start[0], stop[0]), (start[1], stop[1]), markersize=2, color="white")  # plot line segment
# plt.xlabel("x (pixel)")
# plt.ylabel("y (pixel)")
# plt.show()