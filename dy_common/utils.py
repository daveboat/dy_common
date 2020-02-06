import re


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