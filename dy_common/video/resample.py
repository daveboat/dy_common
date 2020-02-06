"""
Resample videos to 30 fps (or some other framerate. If the video is already lower than 30 fps, then just leave it at
its native framerate. This is because, when going from a lower to a higher framerate, ffmpeg often just fills in the
gaps with copies of the previous frame. For example, when going from 15 fps to 30 fps, we'll get [frame 1] [frame 1]
[frame 2] [frame 2] ... and so forth. This messes up the optical flow, so we want the option to check the fps and if it's
lower than the target fps, just move it to the output folder instead of resampling it
"""
import os
import subprocess
import tqdm
import shutil


def checkfps(infile):
    """
    Uses ffprobe to check the frames per second of an input video

    infile: path to video file (full path to be safe)
    """
    foo = subprocess.check_output(
        ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
         infile]).decode("utf-8").rstrip()
    values = foo.split('/')
    return int(values[0]) / int(values[1])


if __name__ == '__main__':

    clips_directory = '../../UCF101'
    out_folder = '../../UCF101_30FPS'
    target_fps = 30

    # make the output folder if it doesn't exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # use the ffmpeg command ffmpeg -y -i <input> -filter:v fps=fps=30 <output>
    files = os.listdir(clips_directory)
    numfiles = len(files)
    copied = 0
    resampled = 0
    for file in tqdm.tqdm(files):
        # check framerate. if framerate is over target, then resample. otherwise, just copy the file over
        if checkfps(os.path.join(clips_directory, file)) > target_fps:
            resampled += 1
            subprocess.call(['ffmpeg', '-y', '-i', os.path.join(clips_directory, file), '-filter:v', 'fps=fps='+str(target_fps), os.path.join(out_folder, file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            copied += 1
            shutil.copy(os.path.join(clips_directory, file), out_folder)
    print('copied %d and resampled %d' %(copied, resampled))