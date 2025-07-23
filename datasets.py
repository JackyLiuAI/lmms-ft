import av
import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "llava-onevision": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
    "llama-3.2-vision": True,
}


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        data_path: str, 
        model_family_id: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
        image_max_size: Optional[int] = None,
        image_min_size: Optional[int] = None,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = TO_LOAD_IMAGE[model_family_id]
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.image_max_size = image_max_size
        self.image_min_size = image_min_size

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")

            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                img = Image.open(image_path).convert("RGB") if self.load_image else image_path

                # 自动调整图片尺寸
                if self.load_image and isinstance(img, Image.Image):
                    width, height = img.size
                    max_size = self.image_max_size
                    min_size = self.image_min_size
                    resize_needed = False

                    # 第一步：等比例缩放，保证最长边不大于max_size
                    if max_size is not None:
                        if width > max_size or height > max_size:
                            # 计算缩放比例
                            ratio = min(max_size / width, max_size / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                            width, height = new_width, new_height
                            resize_needed = True
                    
                    # 第二步：确保所有边都不小于min_size
                    if min_size is not None:
                        new_width, new_height = width, height
                        
                        # 检查是否有边小于最小尺寸
                        if width < min_size:
                            new_width = min_size
                            resize_needed = True
                        
                        if height < min_size:
                            new_height = min_size
                            resize_needed = True
                        
                        # 如果需要调整大小
                        if new_width != width or new_height != height:
                            img = img.resize((new_width, new_height), Image.LANCZOS)

                images.append(img)

        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)
        
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )