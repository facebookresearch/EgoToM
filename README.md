# EgoToM: Benchmarking Theory of Mind Reasoning from Egocentric Videos \[[paper](http://arxiv.org/abs/2503.22152)\]

EgoToM is a new egocentric theory-of-mind benchmark built on Ego4D videos, containing multi-choice questions that evaluate multimodal large language models' ability to infer a camera wearer's goals, in-the-moment belief states, and future actions.  

### Repo structure 
This repo contains the EgoToM benchmark and example model evaluation code. For VLM evaluation, follow the model-specific open source repo to create a separate environment to evaluate each model.

**`code/`**
* `VLMs/` contains the inference code for different models adapted from the demo code or example inference code in open-source repos.
* `generate_video_context.py` is an example script for cropping the video contexts from the original Ego4D videos based on the query moments.
* `vlm_evaluate.py` supports batch evaluating a VLM on multiple conditions. It take a config file that specifies the prompts, conditions wish to be evaluated, video input directories, frame selection, output dir, etc. (see example config in `config/VLMeval/`).

**`egotom/`** 
* contains a superset of the problems in `egotom_paper/`, with additional questions not covered in the paper, for a total of:
    + 354 action questions
    + 335 belief questions
    + 351 goal questions
* Video contexts can be generated using `code/generate_video_context.py` after obtaining Ego4D videos [here](https://ego4d-data.org/#download).
* `egotom_{question}_shuffled.csv` contains the choices and correct answers for each question. The columns include:
    + `vuid`: Ego4D video UIDs
    + `cuid`: EgoToM clip UIDs, has the format `{vuid}~{narrator}~{clip_narration_start_index}-{clip_narration_end_index}`
    + `narrations_in_context`: the human narrations in the video clip up until the query moment (timestamps are relative to the clip start time)
    + `gt_{question}`: correct answer for the question
    + `{question}_choice_{abcd}` columns indicate the choices
    + `clip_start_time` and `clip_end_time` indicate the clip start and end times relative to the original Ego4D video
* `all_prompts.json` contains the prompts for evaluating VLMs. These prompts are directly generated using columns in `egotom_{question}_shuffled.csv`.
    + first level: question type keys `['goal', 'belief', 'actions']`
    + second level: clip UID `cuid` keys to `{'system': str, 'user': str}` prompt pairs.

**`egotom_paper/`** 
* contains the set of problems used in the PAPER:
    + 267 action questions
    + 202 belief questions
    + 237 goal questions

### Example usage

* After setting up model-specific environments and downloading model weights, try `python code/vlm_evaluate.py --config config/VLMeval/run_evaluation_multiexp.yaml`


### License
The data is released CC-by-NC and is intended for benchmarking purposes only. The data includes outputs from GPT-4 Turbo and is subject to the OpenAI terms
