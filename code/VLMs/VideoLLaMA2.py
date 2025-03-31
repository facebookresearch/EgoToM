# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from decord import VideoReader, cpu
from videollama2.model import load_pretrained_model
from videollama2.mm_utils import tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama2.constants import DEFAULT_VIDEO_TOKEN
from utils import select_video_frames

# inference code adapted from https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/

def load_video(video_path, frame_selection):

    if frame_selection == 'none':
        return None

    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vreader)
    selected_frames_ind = select_video_frames(total_frames, selection=frame_selection)
    video_data = vreader.get_batch(selected_frames_ind)
    return video_data

def load_model(variant):
    if variant == 'VideoLLaMA2-7B':
        model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    elif variant == 'VideoLLaMA2-7B-16F':
        model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-16F'
    elif variant == 'VideoLLaMA2-72B':
        model_path = 'DAMO-NLP-SG/VideoLLaMA2-72B'
    else:
        raise ValueError('unsupported model variant')

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    return model, processor, tokenizer

def single_query(prompt, video_data, model, processor, tokenizer, temperature):

    if video_data is not None:
        modal_token = DEFAULT_VIDEO_TOKEN
        images = [Image.fromarray(frame) for frame in video_data.asnumpy()]
        video_input = processor.preprocess(images, return_tensors='pt')['pixel_values']
        tensor = video_input.half().cuda()
        tensor = [(tensor, 'video')]
        # prepend a <video> token to QA prompt as in the official inference code
        prompt['user'] = modal_token + '\n' + prompt['user']
    else:
        modal_token = ''
        tensor = None

    message = [
        {'role': 'system', 'content': prompt['system']},
        {'role': 'user', 'content': prompt['user']}
    ]
    message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(message, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=False,
            temperature=temperature,
            max_new_tokens=200,
            top_p=0.9,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

class VideoLLaMA2():

    def __init__(self, variant):

        self.variant = variant
        model, processor, tokenizer = load_model(variant)
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    
    def query(self, prompt, video_path, frame_selection, video_size, temperature):
    
        try:
            video_data = load_video(video_path=video_path, frame_selection=frame_selection)
        except Exception as error:
            print(f'cannot read video', error)
            return None
        response = single_query(
            prompt=prompt, 
            video_data=video_data, 
            model=self.model, 
            processor=self.processor, 
            tokenizer=self.tokenizer,
            temperature=temperature
        )
        return response
