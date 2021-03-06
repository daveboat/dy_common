import re
import numpy as np


def blank(x, *args, **kwargs):
    """
    A blank function which just returns its first argument, and can take any number of other arguments or keyword
    arguments.
    """
    return x


def parse_sample_distribution(sample_distribution_string):
    """
    Takes a string in the form 'int,int,int,int, ...', and parses it into a tuple (int, int, int, int, ...)
    """

    sample_distribution_string = sample_distribution_string.strip()
    sample_distribution = sample_distribution_string.split(',')
    try:
        sample_distribution = [int(x) for x in sample_distribution]
    except:
        print('Unable to parse sample distribution {}'.format(sample_distribution_string))
        sample_distribution = []

    return tuple(sample_distribution)


def loguniform(low=0, high=1, size=None, base=np.e):
    """
    Works like np.random.uniform, but with a log-uniform distribution
    """
    return np.power(base, np.random.uniform(low, high, size))


def convert_seconds_to_hms(seconds):
    """
    Convert seconds to HH:mm:ss
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def tryint(s):
    """
    int() with exception handling
    """
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def IoU(detection1, detection2):
    """
    Compute the IoU for two bounding boxes

    detection1 and detection2 are (x1, y1, x2, y2, ...), where x and y are in normalized coordinates

    returns the intersection over union between the two bounding boxes
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(detection1[0], detection2[0])
    yA = max(detection1[1], detection2[1])
    xB = min(detection1[2], detection2[2])
    yB = min(detection1[3], detection2[3])

    # area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (detection1[3] - detection1[1]) * (detection1[2] - detection1[0])
    boxBArea = (detection2[3] - detection2[1]) * (detection2[2] - detection2[0])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou