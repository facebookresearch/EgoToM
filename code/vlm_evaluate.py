# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from utils import (
    read_yaml, 
    save_txt, 
    read_json,
)

def load_model(model, temperature):
    if model == 'CogVLM2':
        from VLMs.CogVLM2 import CogVLM2
        return CogVLM2(load='full')
    elif model == 'gpt-4-turbo':
        from VLMs.OpenAI import MultimodalGPT
        return MultimodalGPT(model='gpt-4-turbo', temperature=temperature)
    elif 'VideoLLaMA2' in model:
        from VLMs.VideoLLaMA2 import VideoLLaMA2
        return VideoLLaMA2(variant=model)
    else:
        raise ValueError(f'{model} is unsupported atm.')

def batch_query(
        prompts, 
        uids, 
        video_paths, 
        model, 
        frame_selection,
        video_size,
        temperature=0.0,
        save_dir=None
    ):

    print(f'... collecting N = {len(prompts)} responses ...')
    
    for id, prompt, video_path in zip(uids, prompts, video_paths):
        
        try:
            response_str = model.query(
                prompt=prompt,
                video_path=video_path, 
                frame_selection=frame_selection, 
                video_size=video_size,
                temperature=temperature
            )
            if response_str is None:
                continue
            fname = f'{save_dir}/{id}.txt'
            save_txt(response_str, fname)
        
        except Exception as error:
            print(f'cannot query {id}:', error)
    
def run_eval(config):

    output_dir = config['output_dir']
    model_name = config['model']
    exp_name = config['exp_name']

    ALL_PROMPTS = read_json(config['input_prompts'])
    model = load_model(model_name, config['temperature'])

    for condition in config['conditions']:
        for question in config['questions']:

            target_prompts = ALL_PROMPTS[question]
            frame_selection = config['frame_selection'][condition]

            # set up exp dir
            exp_dir = f'{output_dir}/{model_name}_{exp_name}-{question}-{condition}-{frame_selection}'
            config['exp_dir'] = exp_dir
            if not os.path.exists(exp_dir):
                os.mkdir(exp_dir)
            print(f'\nExperiment dir: {exp_dir}...')

            # screen out already-answered questions
            uids = target_prompts.keys()
            prompts_new, uids_new = [], []
            video_paths = []
            for uid in uids:
                if os.path.exists(f'{exp_dir}/{uid}.txt'): continue
                uids_new.append(uid)
                prompts_new.append(target_prompts[uid])
                video_path = config['video_input_dir'][condition] + f'/{uid}_context.mp4'
                video_paths.append(video_path)

            # get model responses
            batch_query(
                prompts=prompts_new,
                uids=uids_new,
                video_paths=video_paths,
                model=model,
                frame_selection=frame_selection,
                video_size=tuple(config['video_size']),
                save_dir=exp_dir,
                temperature=config['temperature']
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    config = read_yaml(args.config)
    run_eval(config=config)
