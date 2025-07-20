#!/bin/bash
# 下载数据集
huggingface-cli download ShareGPT4Video/ShareGPT4Video zip_folder/ego4d/ego4d_videos_4.zip --repo-type dataset --local-dir ./example_data/videos --resume-download
# 解压文件
unzip example_data/videos/zip_folder/ego4d/ego4d_videos_4.zip -d example_data/videos/ego4d

# 清理压缩文件
rm example_data/videos/zip_folder/ego4d/ego4d_videos_4.zip

echo "数据集下载和解压完成！"