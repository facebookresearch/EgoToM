import os
import json
import yaml
import numpy as np

def read_json(fname):
    assert os.path.exists(fname)
    with open(fname) as f:
        x = json.load(f)
        f.close()
    return x

def read_yaml(fname):
    assert os.path.exists(fname)
    with open(fname) as f:
        x = yaml.safe_load(f)
        f.close()
    return x

def save_txt(content, fname):
    with open(fname, 'w') as f:
        f.write(content)
        f.close()

def select_video_frames(total_frames, selection):
    '''
    returns indices of video frames depending on frame selection specification
    '''

    if 'uniform' in selection:
        num_frames = int(selection.split('uniform')[1])
        # hacky uniform, biases selecting the last frame (end-aligned)
        selected_frames = np.arange(
            total_frames-1, # end frame
            -1, # last element is frame 0
            -(total_frames//num_frames) # decrement interval
        )[:num_frames]
        selected_frames = np.flip(selected_frames) # reverse back to time order
        return selected_frames
    elif 'first' in selection:
        num_frames = int(selection.split('first')[1])
        selected_frames = np.arange(0, total_frames)[:num_frames]
        return selected_frames
    elif 'last' in selection:
        num_frames = int(selection.split('last')[1])
        selected_frames = np.arange(0, total_frames)[-num_frames:]
        return selected_frames
    elif selection == 'none':
        return None
    else:
        raise ValueError(f'Unknown frame selection spec! : {selection}')