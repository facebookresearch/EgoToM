# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from torchvision.transforms.functional import resize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from utils import select_video_frames

# modified from https://github.com/THUDM/CogVLM2/tree/main/video_demo/inference.py
# and https://github.com/THUDM/CogVLM2/blob/main/basic_demo/cli_demo_multi_gpus.py

MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def load_video(video_path, frame_selection, video_size):
    '''
    loads a video into a tensor

    args
    ----
    video_path : str, str to raw video file
    frame_selection : str, how to select frames from video. 
        one of ['uniform{int}', 'last{int}', 'first{int}'], where 
        int represents the desired number of frames

    return
    ------
    video_data : torch.tensor, shape (3, n_frame, height, width)
    '''
    if frame_selection == 'none':
        return None

    bridge.set_bridge('torch')
    with open(video_path, 'rb') as f:
        mp4_stream = f.read()

    if mp4_stream is not None:
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    else:
        decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    selected_frames_ind = select_video_frames(total_frames, frame_selection)
    video_data = decord_vr.get_batch(selected_frames_ind)
    video_data = video_data.permute(3, 0, 1, 2)

    video_data = resize(video_data, video_size)
    return video_data

def load_model(load):
    '''
    args
    ----
    load : str, indicates how to load the model. one of ['quant8', 'full']
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        # padding_side="left"
    )
    if load == 'quant8':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=TORCH_TYPE,
            ),
            low_cpu_mem_usage=True
        ).eval().to(DEVICE)
    elif load == 'full':
        # detect gpu memory size and decide if we need multi-gpu handle
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30 * 1024 ** 3:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True
            ).eval().to(DEVICE)
        else: # multi-gpu
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=TORCH_TYPE,
                    trust_remote_code=True,
                )
            num_gpus = torch.cuda.device_count()
            max_memory_per_gpu = "16GiB"
            if num_gpus > 2:
                max_memory_per_gpu = f"{round(42 / num_gpus)}GiB"
            print(f'... gpu memory is less than 30GB. loading model in multi-gpu mode with max {max_memory_per_gpu}/gpu...')

            device_map = infer_auto_device_map(
                model=model,
                max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
                no_split_module_classes=["CogVLMDecoderLayer"]
            )
            # load_checkpoint_and_dispatch wants the absolute path...
            from constants import ABS_MODEL_PATH
            model = load_checkpoint_and_dispatch(model, ABS_MODEL_PATH['CogVLM2'], device_map=device_map, dtype=TORCH_TYPE)
            model = model.eval()
    else:
        raise ValueError(f'Unknown model loading mode: {load}')
    
    return tokenizer, model

def single_query(prompt, video_data, tokenizer, model, temperature=0.0):

    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=prompt,
        images=[video_data] if video_data is not None else None,
        history=[],
        template_version='chat'
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]] if inputs['images'] is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class CogVLM2():

    def __init__(self, load='full'):

        self.tokenizer, self.model = load_model(load)

    def query(self, prompt, video_path, frame_selection, video_size, temperature):

        video_data = load_video(video_path, frame_selection, video_size)
        response = single_query(prompt, video_data, self.tokenizer, self.model, temperature=temperature)
        return response
