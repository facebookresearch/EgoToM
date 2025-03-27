import os
import cv2
import base64
from utils import select_video_frames
from langchain_openai import AzureChatOpenAI

# https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
# https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o

def get_api_credentials():
    return {
        'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
        'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'azure_deployment': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_GPT4T')
    }

def read_video_base64_frames(video_path):
    video = cv2.VideoCapture(video_path)

    base64_frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64_frames), "frames read.")
    return base64_frames

def load_video(video_path, frame_selection, video_size):
    if frame_selection == 'none':
        return []
    base64_frames = read_video_base64_frames(video_path)
    selected_frames_ind = select_video_frames(total_frames=len(base64_frames), selection=frame_selection)
    selected_frames = [f for i, f in enumerate(base64_frames) if i in selected_frames_ind]
    return selected_frames

class MultimodalGPT():

    def __init__(self, model, temperature):

        assert model in ['gpt-4-turbo']
        credentials = get_api_credentials()
        self.llm = AzureChatOpenAI(
            **credentials,
            api_version='2024-05-01-preview',
            temperature=temperature,
            max_tokens=200,
            timeout=None,
            max_retries=2,
        )
    
    def merge_prompt_w_frames(self, prompt, video_frames):
        # back out message list from langchain so we can interleave video frames with QA
        messages = [
            {
                'role': 'system',
                'content': prompt['system']
            },
            {
                'role': 'human',
                'content': [
                    prompt['user'].split('\n')[0], # 'Analyze the video of a human actor C'
                    * map(lambda x: {'type': 'image_url', 'image_url': {'url': f'data:image/jpg;base64,{x}'}}, video_frames),
                    '\n'.join(prompt['user'].split('\n')[2:]) # question and choices
                ]
            }
        ]
        return messages

    def query(self, prompt, video_path, frame_selection, video_size, temperature):

        video_frames = load_video(video_path, frame_selection, video_size)
        if len(video_frames) == 0: # nocontext
            response = self.llm.invoke(prompt)
        else:
            messages = self.merge_prompt_w_frames(prompt, video_frames)
            response = self.llm.invoke(messages)

        return response.content