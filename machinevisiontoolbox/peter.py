from pathlib import Path
import numpy as np

# doesnt handle wild card in name, but just add re matching and linear search through dic

def colorname(arg):

    colordict = {}

    # read the data
    rgbfilename = (Path.cwd() / 'data' / 'rgb.txt').as_posix()
    with open(rgbfilename, "r") as f:
        for line in f:
            if len(line) == 1 or line.startswith('#'):
                continue
            data = line.split()
            colordict[data[3].lower()] = np.r_[[int(x) for x in data[0:3]]] # save as np array

    if isinstance(arg, str) or isinstance(arg, (list, tuple)) and isinstance(arg[0], str):
        # map name to RGB tuple
        if isinstance(arg, str):
            return tuple(colordict[arg.lower()])
        else:
            return [tuple(colordict[x.lower()]) for x in arg]

    elif isinstance(arg, (tuple, list, np.ndarray)):
        # map RGB tuple to name
        mindist = 4 * 255 ** 2
        minnmae = None
        arg = np.r_[arg]  # convert passed tuple to np array
        # TODO should also convert it from  xy or ab to rgb for comparison
        for k, v in colordict.items():
            if np.all(v == arg):
                return k
            else:
                dist = np.linalg.norm(v - arg)
                if dist < mindist:
                    mindist = dist
                    minname = k
        return minname
    else:
        raise TypeError('unknown type')

print(colorname('red'))
print(colorname([100, 200, 50]))