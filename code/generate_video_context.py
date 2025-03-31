# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize

Ego4D_video_dir = '' # original Ego4D video directory
VIDEO_SIZE = [480, 720]

def clip_video(fin, fout, start_time, end_time):
    if not os.path.exists(fin):
        print(f'Cannot find original video at {fin}...')
        return
    
    tmp_fout = 'tmp_clip.mp4'
    ffmpeg_extract_subclip(
        filename=fin, t1=start_time, t2=end_time, targetname=tmp_fout
    )
    ffmpeg_resize(
        video=tmp_fout, output=fout, size=VIDEO_SIZE
    )
    print(f'Clip saved as {fout}')

def generate_video_context_for_qa(
        qa_trials,
        save_dir,
        condition,
    ):
    '''
    clips video context out of the original video for each qa trial

    args
    ----
    qa_trials : big dataframe
    save_dir : str, where to output the video
    condition : str, whether to truncate the video context
        one of 'fullcontext', 'last30sec', 'last5sec'
    '''

    for _, row in qa_trials.iterrows():
        vuid = row['vuid'] # video UIDs
        cuid = row['cuid'] # EgoToM clip UIDs
        
        end_time = row['clip_end_time']
        if condition == 'fullcontext':
            start_time = row['clip_start_time']
        elif condition == 'last30sec':
            assert 'last30sec' in save_dir
            start_time = end_time - 30.0
        elif condition == 'last5sec':
            assert 'last5sec' in save_dir
            start_time = end_time - 5.0
        else:
            raise ValueError('unknown video context condition')

        if not os.path.exists(f'{save_dir}/{cuid}_context.mp4'):
            clip_video(
                fin=f'{Ego4D_video_dir}/{vuid}.mp4',
                fout=f'{save_dir}/{cuid}_context.mp4',
                start_time=start_time,
                end_time=end_time
            )

qa_shuffled = pd.concat([
    pd.read_csv('egotom/egotom_goal_shuffled.csv'),
    pd.read_csv('egotom/egotom_belief_shuffled.csv'),
    pd.read_csv('egotom/egotom_actions_shuffled.csv'),
])
qa_shuffled = qa_shuffled[['vuid', 'cuid', 'clip_start_time', 'clip_end_time']].drop_duplicates()

generate_video_context_for_qa(
    qa_trials=qa_shuffled, 
    save_dir='egotom/videos/last30sec',
    condition='last30sec'
)
