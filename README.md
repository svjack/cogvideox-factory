# CogVideoX Factory 🧪

[中文阅读](./README_zh.md)

Fine-tune Cog family of video models for custom video generation under 24GB of GPU memory ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
</tr>
</table>

## Mochi-1 Step
- installtion
```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
git clone https://github.com/svjack/cogvideox-factory
cd cogvideox-factory
pip install -r requirements.txt
cd training/mochi-1
pip install -r requirements.txt
pip install Pillow==9.5.0
pip install git+https://github.com/huggingface/diffusers.git
huggingface-cli download \
  --repo-type dataset sayakpaul/video-dataset-disney-organized \
  --local-dir video-dataset-disney-organized
bash prepare_dataset.sh
```
- train.sh
```bash
bash train.sh
```

```bash
#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

DATA_ROOT="videos_prepared"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-lora"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python text_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --pin_memory \
  --caption_dropout 0.1 \
  --max_train_steps 2000 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --enable_model_cpu_offload \
  --optimizer adamw \
  --allow_tf32 \
  --report_to None"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"

```

## Tuned inference 
```python
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype = torch.float16)
pipe.load_lora_weights("svjack/mochi_mickey_mice_early_lora")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

i = 50
generator = torch.Generator("cpu").manual_seed(i) 
pipeline_args = {
        "prompt": "A black and white animated scene unfolds with two anthropomorphic Mickey Mice sitting at a table, each holding a glass of wine. The musical notes and symbols float around them, suggesting a playful environment. One Mickey leans forward in curiosity, while the other remains still, sipping his drink. The dynamics shift as one Mickey raises his glass in a toast, and the other responds by clinking glasses. The scene captures the camaraderie and enjoyment between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.",
        "guidance_scale": 6.0,
        "num_inference_steps": 64,
        "height": 480,
        "width": 848,
        "max_sequence_length": 256,
        "output_type": "np",
        "num_frames": 19,
        "generator": generator
    }
    
video = pipe(**pipeline_args).frames[0]
export_to_video(video, "black_white_drinking_scene.mp4")
from IPython import display 
display.clear_output(wait = True)
display.Video("black_white_drinking_scene.mp4")
```


https://github.com/user-attachments/assets/dc28cead-3fd0-4dc7-a5d9-922395c9e513


# LTX-Video
# Finetrainers Setup and Usage Guide

This guide provides instructions for setting up and using the `finetrainers` repository to process video datasets and train models.

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
```

## Clone the Repository

Clone the `finetrainers` repository and navigate to the directory:

```bash
git clone https://github.com/a-r-r-o-w/finetrainers && cd finetrainers
```

## Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
pip install moviepy==1.0.3
pip install opencv-python
pip install Pillow==9.5.0
pip install "torch>=2.3.1"
pip uninstall torchvision
pip install torchvision
```

## Download Datasets

Download the necessary datasets using `huggingface-cli`:

```bash
huggingface-cli download \
  --repo-type dataset svjack/Genshin-Impact-Cutscenes-with-score-organized \
  --local-dir video-dataset-genshin-impact-cutscenes

huggingface-cli download \
  --repo-type dataset svjack/Genshin-Impact-XiangLing-animatediff-with-score-organized \
  --local-dir video-dataset-genshin-impact-xiangling
```

For more information on the datasets, refer to the [dataset documentation](https://github.com/a-r-r-o-w/finetrainers/blob/main/assets/dataset.md?plain=1).

## Process Video Datasets

Use the provided Python script to process the video datasets:

```python
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

def process_video_dataset(data_root, output_root, target_width, target_height, target_frames):
    # Ensure output directories exist
    os.makedirs(output_root, exist_ok=True)
    videos_dir = os.path.join(output_root, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    # Initialize lists to store prompts and video paths
    prompts = []
    video_paths = []

    # Get list of video files
    video_files = [f for f in os.listdir(data_root) if f.endswith('.mp4')]

    # Process each video file
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(data_root, video_file)
        txt_file = os.path.join(data_root, video_file.replace('.mp4', '.txt'))

        # Read prompt from corresponding txt file
        with open(txt_file, 'r') as f:
            prompt = f.read().strip()
            prompts.append(prompt)

        # Load video
        clip = VideoFileClip(video_path)

        # Resize video to target resolution
        if clip.size[0] != target_width or clip.size[1] != target_height:
            clip = clip.resize((target_width, target_height))

        # Calculate current frame count
        current_frames = int(clip.fps * clip.duration)

        # If target_frames is greater than current_frames, extend the video
        if target_frames > current_frames:
            # Calculate how many times the video needs to be repeated
            repeat_count = target_frames // current_frames
            remaining_frames = target_frames % current_frames

            # Create a list of clips to concatenate
            clips_to_concat = [clip] * repeat_count

            # Add remaining frames if needed
            if remaining_frames > 0:
                remaining_clip = clip.subclip(0, remaining_frames / clip.fps)
                clips_to_concat.append(remaining_clip)

            # Concatenate the clips
            clip = concatenate_videoclips(clips_to_concat)

        # Adjust frame count to meet the requirement
        final_frames = int(clip.fps * clip.duration)
        if final_frames % 4 != 0 and final_frames % 4 != 1:
            # Adjust frame count to the nearest valid frame count
            new_frames = (final_frames // 4) * 4
            if final_frames % 4 > 1:
                new_frames += 1
            clip = clip.subclip(0, new_frames / clip.fps)

        # Save processed video
        output_video_path = os.path.join(videos_dir, video_file)
        clip.write_videofile(output_video_path, codec='libx264')

        # Add relative video path to list
        video_paths.append(os.path.join('videos', video_file))

    # Write prompts to prompt.txt
    with open(os.path.join(output_root, 'prompt.txt'), 'w') as f:
        f.write('\n'.join(prompts))

    # Write video paths to videos.txt
    with open(os.path.join(output_root, 'videos.txt'), 'w') as f:
        f.write('\n'.join(video_paths))

# Example usage
data_root = 'video-dataset-genshin-impact-cutscenes'  # Replace with your dataset path
output_root = 'video-dataset-genshin-impact-cutscenes-processed-32-rec'  # Replace with your desired output path
target_width = 720  # Replace with your target width
target_height = 480  # Replace with your target height
target_frames = 32  # Replace with your target frame count

process_video_dataset(data_root, output_root, target_width, target_height, target_frames)

# Example usage
data_root = 'video-dataset-genshin-impact-xiangling'  # Replace with your dataset path
output_root = 'video-dataset-genshin-impact-xiangling-32-rec'  # Replace with your desired output path
target_width = 512  # Replace with your target width
target_height = 768  # Replace with your target height
target_frames = 32  # Replace with your target frame count

process_video_dataset(data_root, output_root, target_width, target_height, target_frames)
```
- OR
```python
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm

def is_video_file(file_path):
    """检查文件是否为视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in video_extensions

def resize_and_pad(frame, target_width, target_height):
    """
    将帧缩放到目标分辨率，并用黑色填充不足的部分。
    """
    # 获取原始帧的尺寸
    original_height, original_width = frame.shape[:2]

    # 计算原始帧和目标分辨率的宽高比
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # 确定缩放因子和新尺寸
    if aspect_ratio > target_aspect_ratio:
        # 如果帧比目标宽，则基于宽度缩放
        scale = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale)
    else:
        # 如果帧比目标高，则基于高度缩放
        scale = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale)

    # 使用 OpenCV 缩放帧
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # 创建一个黑色背景，尺寸为目标分辨率
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算将缩放后的帧放置在黑色背景上的位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将缩放后的帧放置在黑色背景上
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    return padded_frame

def process_video_dataset(data_root, output_root, target_width, target_height, target_frames):
    """
    处理视频数据集：
    1. 调整视频分辨率并填充。
    2. 调整视频帧数，通过循环填充不足的部分。
    3. 保留相同的文件和目录结构。
    """
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)
    videos_dir = os.path.join(output_root, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    # 初始化列表，用于存储提示和视频路径
    prompts = []
    video_paths = []

    # 获取所有视频文件
    video_files = [f for f in os.listdir(data_root) if is_video_file(os.path.join(data_root, f))]

    # 处理每个视频文件
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(data_root, video_file)
        txt_file = os.path.join(data_root, video_file.replace('.mp4', '.txt'))

        # 检查是否存在对应的提示文件
        if not os.path.exists(txt_file):
            print(f"Warning: No prompt file found for {video_file}. Skipping.")
            continue

        # 读取提示内容
        with open(txt_file, 'r') as f:
            prompt = f.read().strip()
            prompts.append(prompt)

        # 加载视频
        try:
            clip = VideoFileClip(video_path)
        except Exception as e:
            print(f"Error loading video {video_file}: {e}. Skipping.")
            continue

        # 初始化列表，用于存储处理后的帧
        processed_frames = []

        # 计算视频的总帧数
        total_frames = int(clip.fps * clip.duration)

        # 如果目标帧数大于当前帧数，则通过循环填充
        if target_frames > total_frames:
            # 计算需要循环的次数
            repeat_count = target_frames // total_frames
            remaining_frames = target_frames % total_frames

            # 循环视频帧
            for _ in range(repeat_count):
                for frame in clip.iter_frames():
                    processed_frame = resize_and_pad(frame, target_width, target_height)
                    processed_frames.append(processed_frame)

            # 添加剩余的帧
            for i in range(remaining_frames):
                frame = clip.get_frame(i / clip.fps)
                processed_frame = resize_and_pad(frame, target_width, target_height)
                processed_frames.append(processed_frame)
        else:
            # 如果目标帧数小于等于当前帧数，则直接截取
            frame_step = max(1, total_frames // target_frames)
            for i, frame in enumerate(clip.iter_frames()):
                if i % frame_step == 0:
                    processed_frame = resize_and_pad(frame, target_width, target_height)
                    processed_frames.append(processed_frame)
                if len(processed_frames) >= target_frames:
                    break

        # 从处理后的帧创建新视频
        processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)

        # 保存处理后的视频
        output_video_path = os.path.join(videos_dir, video_file)
        processed_clip.write_videofile(output_video_path, codec='libx264')

        # 添加相对视频路径到列表
        video_paths.append(os.path.join('videos', video_file))

    # 将提示写入 prompt.txt
    with open(os.path.join(output_root, 'prompt.txt'), 'w') as f:
        f.write('\n'.join(prompts))

    # 将视频路径写入 videos.txt
    with open(os.path.join(output_root, 'videos.txt'), 'w') as f:
        f.write('\n'.join(video_paths))

# Example usage
data_root = 'video-dataset-genshin-impact-xiangling'  # Replace with your dataset path
output_root = 'video-dataset-genshin-impact-xiangling-49-rec-pad'  # Replace with your desired output path
target_width = 768  # Replace with your target width
target_height = 512  # Replace with your target height
target_frames = 49  # Replace with your target frame count

process_video_dataset(data_root, output_root, target_width, target_height, target_frames)
```

## Upload dataset to hub 
```python
import os
import shutil
import pandas as pd
from huggingface_hub import HfApi

# 输入路径
input_folder = "video-dataset-genshin-impact-xiangling-49-rec-pad-cp/"
# 新建路径用于文件准备
output_folder = "video-dataset-prepared"

# 确保新建路径存在
os.makedirs(output_folder, exist_ok=True)

# 读取 prompts 和 videos
with open(os.path.join(input_folder, "prompt.txt"), "r") as f:
    prompts = f.read().splitlines()
with open(os.path.join(input_folder, "videos.txt"), "r") as f:
    video_paths = f.read().splitlines()

# 创建 videos 文件夹
videos_dir = os.path.join(output_folder, "videos")
os.makedirs(videos_dir, exist_ok=True)

# 将 prompts 和 video_paths 转换为字典，方便查找
prompt_dict = {}
for video_path, prompt in zip(video_paths, prompts):
    video_filename = os.path.basename(video_path)
    prompt_dict[video_filename] = prompt

# 复制视频文件到新路径，并准备 metadata.csv 数据
metadata_data = []
for video_path in video_paths:
    # 视频文件名
    video_filename = os.path.basename(video_path)
    # 检查是否有对应的提示
    if video_filename not in prompt_dict:
        print(f"警告: 视频文件 {video_filename} 没有对应的提示，已跳过。")
        continue
    # 源视频路径
    src_video_path = os.path.join(input_folder, video_path)
    # 目标视频路径
    dst_video_path = os.path.join(videos_dir, video_filename)
    # 复制视频文件
    shutil.copy(src_video_path, dst_video_path)
    # 添加到 metadata.csv 数据（在 file_name 中加上 videos/ 路径）
    metadata_data.append({"file_name": f"videos/{video_filename}", "text": prompt_dict[video_filename]})

metadata = pd.DataFrame(metadata_data)
metadata.to_csv(os.path.join(output_folder, "metadata.csv"), index=False) 

print("文件准备完成，路径:", output_folder)

repo_id = "svjack/Genshin-Impact-XiangLing-animatediff-hf"  # 替换为你的数据集名称
api = HfApi()
api.upload_folder(
    folder_path=output_folder,
    repo_id=repo_id,
    repo_type="dataset",
)

print(f"数据集已上传到 Hugging Face Hub: {repo_id}")
```


## Download Pretrained Model

Download the pretrained model from Hugging Face:

```bash
git clone https://huggingface.co/Lightricks/LTX-Video
```

Alternatively, use `huggingface-cli`:

```bash
huggingface-cli download \
  --repo-type model Lightricks/LTX-Video \
  --local-dir LTX-Video
```
-OR 
```python
import os 
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
# 下载模型
repo_id = "Lightricks/LTX-Video"
local_dir = "LTX-Video"
# 使用 snapshot_download 下载模型
snapshot_download(
    repo_id=repo_id,
    repo_type="model",  # 指定仓库类型为模型
    local_dir=local_dir,  # 指定本地保存目录
    local_dir_use_symlinks=False,  # 不使用符号链接
    resume_download=True,  # 支持断点续传
)
```

## Run the Training Script

Create and execute a shell script to run the training process:

```bash
vim run_xiangling.sh
```

Add the following content to the script:

```bash
#!/bin/bash

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0"

DATA_ROOT="video-dataset-genshin-impact-xiangling-49-rec-pad"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="ltxv_xiangling_save"

# Model arguments
model_cmd="--model_name ltx_video \
  --pretrained_model_name_or_path LTX-Video"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --video_resolution_buckets 49x512x768 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 0"

# Diffusion arguments
diffusion_cmd="--flow_resolution_shifting"

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 15000 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --checkpointing_limit 5 \
  --enable_slicing --precompute_conditions \
  --enable_tiling"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-ltxv \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to None"

#### Use config file in https://github.com/svjack/cogvideox-factory/accelerate_configs/uncompiled_1.yaml
cmd="accelerate launch --config_file accelerate_configs/uncompiled_1.yaml train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $diffusion_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

Make the script executable and run it:

```bash
chmod +x run_xiangling.sh
./run_xiangling.sh
```

```bash
huggingface-cli download \
  --repo-type model hunyuanvideo-community/HunyuanVideo \
  --local-dir HunyuanVideo
```

```bash
#!/bin/bash

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0"

DATA_ROOT="video-dataset-genshin-impact-xiangling-49-rec-pad"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="hunyuan_xiangling_save"

# Model arguments
model_cmd="--model_name hunyuan_video \
  --pretrained_model_name_or_path HunyuanVideo"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --video_resolution_buckets 17x512x768 49x512x768 61x512x768 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 0"

# Diffusion arguments
diffusion_cmd="--flow_resolution_shifting"

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 15000 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --checkpointing_limit 5 \
  --enable_slicing --precompute_conditions \
  --enable_tiling"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-hunyuan \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to None"

#### Use config file in https://github.com/svjack/cogvideox-factory/accelerate_configs/uncompiled_1.yaml
cmd="accelerate launch --config_file accelerate_configs/uncompiled_1.yaml train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $diffusion_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

## Conclusion

This guide has walked you through the setup, dataset processing, and training process for the `finetrainers` repository. For further details, refer to the repository's documentation and the provided scripts.


## Quickstart

Clone the repository and make sure the requirements are installed: `pip install -r requirements.txt` and install diffusers from source by `pip install git+https://github.com/huggingface/diffusers`.
```bash
pip install -r requirements.txt
pip install huggingface_hub accelerate
pip install git+https://github.com/huggingface/diffusers
```

Then download a dataset:

```bash
# install `huggingface_hub`
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
```

Then launch LoRA fine-tuning for text-to-video (modify the different hyperparameters, dataset root, and other configuration options as per your choice):

```bash
# For LoRA finetuning of the text-to-video CogVideoX models
./train_text_to_video_lora.sh

# For full finetuning of the text-to-video CogVideoX models
./train_text_to_video_sft.sh

# For LoRA finetuning of the image-to-video CogVideoX models
./train_image_to_video_lora.sh
```

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import CogVideoXPipeline
from diffusers import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

**Note:** For Image-to-Video finetuning, you must install diffusers from [this](https://github.com/huggingface/diffusers/pull/9482) branch (which adds lora loading support in CogVideoX image-to-video) until it is merged.

Below we provide additional sections detailing on more options explored in this repository. They all attempt to make fine-tuning for video models as accessible as possible by reducing memory requirements as much as possible.

## Prepare Dataset and Training

Before starting the training, please check whether the dataset has been prepared according to the [dataset specifications](assets/dataset.md). We provide training scripts suitable for text-to-video and image-to-video generation, compatible with the [CogVideoX model family](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce). Training can be started using the `train*.sh` scripts, depending on the task you want to train. Let's take LoRA fine-tuning for text-to-video as an example.

- Configure environment variables as per your choice:

  ```bash
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- Configure which GPUs to use for training: `GPU_IDS="0,1"`

- Choose hyperparameters for training. Let's try to do a sweep on learning rate and optimizer type as an example:

  ```bash
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("3000")
  ```

- Select which Accelerate configuration you would like to train with: `ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`. We provide some default configurations in the `accelerate_configs/` directory - single GPU uncompiled/compiled, 2x GPU DDP, DeepSpeed, etc. You can create your own config files with custom settings using `accelerate config --config_file my_config.yaml`.

- Specify the absolute paths and columns/files for captions and videos.

  ```bash
  DATA_ROOT="/path/to/my/datasets/video-dataset-disney"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- Launch experiments sweeping different hyperparameters:
  ```
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/path/to/my/models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_text_to_video_lora.py \
            --pretrained_model_name_or_path THUDM/CogVideoX-5b \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --id_token BW_STYLE \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 10 \
            --seed 42 \
            --rank 128 \
            --lora_alpha 128 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --max_train_steps $steps \
            --checkpointing_steps 1000 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 400 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --max_grad_norm 1.0 \
            --allow_tf32 \
            --report_to wandb \
            --nccl_timeout 1800"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

  To understand what the different parameters mean, you could either take a look at the [args](./training/args.py) file or run the training script with `--help`.

Note: Training scripts are untested on MPS, so performance and memory requirements can differ widely compared to the CUDA reports below.

## Memory requirements

<table align="center">
<tr>
  <td align="center" colspan="2"><b>CogVideoX LoRA Finetuning</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/lora_2b.png" /></td>
  <td align="center"><img src="assets/lora_5b.png" /></td>
</tr>

<tr>
  <td align="center" colspan="2"><b>CogVideoX Full Finetuning</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/sft_2b.png" /></td>
  <td align="center"><img src="assets/sft_5b.png" /></td>
</tr>
</table>

Supported and verified memory optimizations for training include:

- `CPUOffloadOptimizer` from [`torchao`](https://github.com/pytorch/ao). You can read about its capabilities and limitations [here](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload). In short, it allows you to use the CPU for storing trainable parameters and gradients. This results in the optimizer step happening on the CPU, which requires a fast CPU optimizer, such as `torch.optim.AdamW(fused=True)` or applying `torch.compile` on the optimizer step. Additionally, it is recommended not to `torch.compile` your model for training. Gradient clipping and accumulation is not supported yet either.
- Low-bit optimizers from [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/optimizers). TODO: to test and make [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) ones work
- DeepSpeed Zero2: Since we rely on `accelerate`, follow [this guide](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed) to configure your `accelerate` installation to enable training with DeepSpeed Zero2 optimizations. 

> [!IMPORTANT]
> The memory requirements are reported after running the `training/prepare_dataset.py`, which converts the videos and captions to latents and embeddings. During training, we directly load the latents and embeddings, and do not require the VAE or the T5 text encoder. However, if you perform validation/testing, these must be loaded and increase the amount of required memory. Not performing validation/testing saves a significant amount of memory, which can be used to focus solely on training if you're on smaller VRAM GPUs.
>
> If you choose to run validation/testing, you can save some memory on lower VRAM GPUs by specifying `--enable_model_cpu_offload`.

### LoRA finetuning

> [!NOTE]
> The memory requirements for image-to-video lora finetuning are similar to that of text-to-video on `THUDM/CogVideoX-5b`, so it hasn't been reported explicitly.
>
> Additionally, to prepare test images for I2V finetuning, you could either generate them on-the-fly by modifying the script, or extract some frames from your training data using:
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`,
> or provide a URL to a valid and accessible image.

<details>
<summary> AdamW </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.764          |         46.918          |       24.234         |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.121          |       24.234         |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.314          |         47.469          |       24.469         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          13.035          |         21.564          |       24.500         |
| THUDM/CogVideoX-2b |    256    |          False         |         13.095         |          45.826          |         48.990          |       25.543         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          13.095          |         22.344          |       25.537         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.746          |       38.123         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         30.338          |       38.738         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          22.119          |         31.939          |       41.537         |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.803          |         21.814          |       24.322         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          22.254          |         22.254          |       24.572         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.033          |       25.574         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.492          |         46.492          |       38.197         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          47.805          |         47.805          |       39.365         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |       41.008         |

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.732          |         46.887          |        24.195        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.430          |        24.195        |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.004          |         47.158          |        24.369        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         21.297          |        24.357        |
| THUDM/CogVideoX-2b |    256    |          False         |         13.035         |          45.291          |         48.455          |        24.836        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.035         |          13.035          |         21.625          |        24.869        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.602          |        38.049        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         29.359          |        38.520        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          21.352          |         30.727          |        39.596        |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.734          |         21.775          |       24.281         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          21.941          |         21.941          |       24.445         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.266          |       24.943         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.320          |         46.326          |       38.104         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.820          |         46.820          |       38.588         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.920          |         47.980          |       40.002         |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer (with gradient offloading) </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.705          |         46.859          |       24.180         |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.395          |       24.180         |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          43.916          |         47.070          |       24.234         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         20.887          |       24.266         |
| THUDM/CogVideoX-2b |    256    |          False         |         13.095         |          44.947          |         48.111          |       24.607         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.095         |          13.095          |         21.391          |       24.635         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.533          |       38.002         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.006          |         29.107          |       38.785         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          20.771          |         30.078          |       39.559         |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.709          |         21.762          |       24.254         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          21.844          |         21.855          |       24.338         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.031          |       24.709         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.262          |         46.297          |       38.400         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.561          |         46.574          |       38.840         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |       39.623         |

</details>

<details>
<summary> DeepSpeed (AdamW + CPU/Parameter offloading) </summary>

**Note:** Results are reported with `gradient_checkpointing` enabled, running on a 2x A100.

With `train_batch_size = 1`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          13.141          |         21.070          |       24.602         |
| THUDM/CogVideoX-5b |         20.170         |          20.170          |         28.662          |       38.957         |

With `train_batch_size = 4`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          19.854          |         20.836          |       24.709         |
| THUDM/CogVideoX-5b |         20.170         |          40.635          |         40.699          |       39.027         |

</details>

### Full finetuning

> [!NOTE]
> The memory requirements for image-to-video full finetuning are similar to that of text-to-video on `THUDM/CogVideoX-5b`, so it hasn't been reported explicitly.
>
> Additionally, to prepare test images for I2V finetuning, you could either generate them on-the-fly by modifying the script, or extract some frames from your training data using:
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`,
> or provide a URL to a valid and accessible image.

> [!NOTE]
> Trying to run full finetuning without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

<details>
<summary> AdamW </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          33.934          |         43.848          |       37.520         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          OOM             |         OOM             |       OOM            |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          38.281          |         48.341          |       37.544         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          OOM             |         OOM             |       OOM            |

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.447          |         27.555          |       27.156         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          52.826          |         58.570          |       49.541         |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.930          |         27.990          |       27.326         |
| THUDM/CogVideoX-5b |          True          |         16.396         |          66.648          |         66.705          |       48.828         |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer (with gradient offloading) </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.396          |         26.100          |       23.832         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          39.359          |         48.307          |       37.947         |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.916          |         27.975          |       23.936         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          66.607          |         66.668          |       38.061         |

</details>

<details>
<summary> DeepSpeed (AdamW + CPU/Parameter offloading) </summary>

**Note:** Results are reported with `gradient_checkpointing` enabled, running on a 2x A100.

With `train_batch_size = 1`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          13.111          |         20.328          |       23.867         |
| THUDM/CogVideoX-5b |         19.762         |          19.998          |         27.697          |       38.018         |

With `train_batch_size = 4`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          21.188          |         21.254          |       23.869         |
| THUDM/CogVideoX-5b |         19.762         |          43.465          |         43.531          |       38.082         |

</details>

> [!NOTE]
> - `memory_after_validation` is indicative of the peak memory required for training. This is because apart from the activations, parameters and gradients stored for training, you also need to load the vae and text encoder in memory and spend some memory to perform inference. In order to reduce total memory required to perform training, one can choose not to perform validation/testing as part of the training script.
>
> - `memory_before_validation` is the true indicator of the peak memory required for training if you choose to not perform validation/testing.

<table align="center">
<tr>
  <td align="center"><a href="https://www.youtube.com/watch?v=UvRl4ansfCg"> Slaying OOMs with PyTorch</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/slaying-ooms.png" style="width: 480px; height: 480px;"></td>
</tr>
</table>

## TODOs

- [x] Make scripts compatible with DDP
- [ ] Make scripts compatible with FSDP
- [x] Make scripts compatible with DeepSpeed
- [ ] vLLM-powered captioning script
- [x] Multi-resolution/frame support in `prepare_dataset.py`
- [ ] Analyzing traces for potential speedups and removing as many syncs as possible
- [ ] Support for QLoRA (priority), and other types of high usage LoRAs methods
- [x] Test scripts with memory-efficient optimizer from bitsandbytes
- [x] Test scripts with CPUOffloadOptimizer, etc.
- [ ] Test scripts with torchao quantization, and low bit memory optimizers (Currently errors with AdamW (8/4-bit torchao))
- [ ] Test scripts with AdamW (8-bit bitsandbytes) + CPUOffloadOptimizer (with gradient offloading) (Currently errors out)
- [ ] [Sage Attention](https://github.com/thu-ml/SageAttention) (work with the authors to support backward pass, and optimize for A100)

> [!IMPORTANT]
> Since our goal is to make the scripts as memory-friendly as possible we don't guarantee multi-GPU training.
