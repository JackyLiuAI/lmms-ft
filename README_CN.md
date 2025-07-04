# 实现最新mllms的微调

活跃维护者：[Yuqian Hong](https://github.com/lavinal712)

初始维护者：[Jingyang Zhang](https://github.com/zjysteven)、[Yueqian Lin](https://github.com/linyueqian)

## 关于

越来越多的大型多模态模型 (LMM) 不断发布，但这些模型的微调并不总是那么简单。此代码库旨在为 LMM 微调提供统一、精简的结构。关键设计理念包括：

- 微调过程的组件（例如，模型加载、数据整理）被抽象出来，从而允许人们轻松地将最新的 LMM 集成到该代码库中，并以最小的努力对其进行微调；
- 对于所有 LMM，都使用🤗huggingface 的官方实现，以便在微调之后可以以与之前使用 HF 模型完全相同的方式进行推理和其他所有操作；
- 代码库尽可能保持简单/轻量，以便于理解和修改。

该代码库非常灵活。它支持各种类型的 LMM 的微调，包括：

- 🌇 单图像模型：[LLaVA-1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0)、[LLaVA-1.6/NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf)、[Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)、[Llama-3.2-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- 📑 多个/交错图像模型：[Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)、[Qwen2-VL-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)、 [LLaVA-NeXT-Interleave](https://huggingface.co/collections/llava-hf/llava-interleave-668e19a97da0036aad4a2f19)、[Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- 🎥 视频模型：[LLaVA-NeXT-Video](https://huggingface.co/collections/llava-hf/llava-next-video-6666a9173a64c7052930f153)
- 🚀 统一模型：[LLaVA-Onevision](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe)

请参阅[supported_models.md](https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/supported_models.md)，查看受支持模型的完整列表。训练策略方面，LLM 组件支持 1) full-finetuning、2) lora 和 3) q-lora；视觉编码器/主干网支持 1) full-finetuning 和 2) lora。

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: 16px;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">与其他训练框架（例如 LLaMA-Factory、xtuner、swift）有何不同？</font></font></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">这些都是规模庞大、优化程度极高的项目/框架，非常出色。然而，由于其规模和复杂性，它们可能不够透明，上手难度也比较高（例如，我个人在尝试使用这些框架时会感到很迷茫，总是会问自己“应该如何格式化数据”之类的问题）。而这个代码库 (lmms-finetune) 则以轻量级和简洁为设计理念，这意味着您可以更快速地上手，并根据需要了解训练过程的几乎所有细节。换句话说，这是一个极简可行代码库，它支持 LMM 微调，同时方便快速实验、灵活修改以及轻松集成新模型。</font></font></p></details>

## 消息

- **2025/01/27**：此分支支持 Qwen2.5 系列`transformers-4.49.0.dev0`。目前您需要从 GitHub 安装最新的 Transformers。
- **2024/12/16 ：感谢**[lavinal712 (Yuqian)](https://github.com/lavinal712)的贡献，现在支持使用 Llama-3.2-Vision 进行训练。此外，还添加了一个实用脚本`merge_lora_weights.py`。
- **2024年10月16日**：我们添加了LLaVA-Onevision。使用LLaVA-Onevision的注意事项请见[此处](https://github.com/zjysteven/lmms-finetune/issues/43)。此外，我们还更新了排序器，以与Transformer中LLaVA模型的新处理方式保持一致。
- **2024/08/28**：支持使用 gradio webui 界面进行微调。尝试一下`python webui.py`。
- **2024/07/30**：现在支持视觉编码器和投影仪的微调。
- **2024/07/25**：多项改进。我们*1）*发布了一个[Colab 笔记本，](https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing)演示了使用 LLaVA-NeXT-Video-7B 进行完整且成功的训练运行（很高兴听到大家也表示他们[的案例](https://github.com/zjysteven/lmms-finetune/issues/7#issuecomment-2249864887)取得了成功）；*2）*支持在训练集中使用纯文本样本（详情请参阅[此处）。](https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/dataset.md)
- **2024/07/20**：代码库首次发布。更多模型和优化即将推出。敬请期待！

## 安装

```sh
# clone this repo
git clone https://github.com/zjysteven/lmms-finetune.git

# set up a conda environment
conda create -n lmms-finetune python=3.10 -y
conda activate lmms-finetune
## this will install the latest version of torch
## feel free to change it to a specific version
python -m pip install -r requirements.txt

## optionally install flash attention
python -m pip install --no-cache-dir --no-build-isolation flash-attn
```

## 用法

[此Colab 笔记本](https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing)展示了一个可行的训练示例（LLaVA-NeXT-Video-7B），这是一个很好的起点，可以帮助您了解如何使用此代码库。以下部分将提供有关如何微调模型的更详细指南。

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: 16px;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: 600;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">0.查看你想要微调的模型是否受支持</font></font></b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">浏览</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/supported_models.md" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">supported_models.md</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">。或者运行</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">python supported_models.py</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">，例如，它将显示以下内容</font></font></p><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: 16px; background-color: rgb(246, 248, 250);"><pre class="notranslate" style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: 16px; overflow: auto; line-height: 1.45; color: rgb(31, 35, 40); background-color: rgb(246, 248, 250); border-radius: 6px;"><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: rgba(0, 0, 0, 0); border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;">Supported models:
  Model ID                      : HuggingFace Path
  ------------------------------------------------
  llava-1.5-7b                  : llava-hf/llava-1.5-7b-hf
  llava-1.5-13b                 : llava-hf/llava-1.5-13b-hf
  llava-next-video-7b           : llava-hf/LLaVA-NeXT-Video-7B-hf
  llava-next-video-7b-32k       : llava-hf/LLaVA-NeXT-Video-7B-32K-hf
  llava-next-video-34b          : llava-hf/LLaVA-NeXT-Video-34B-hf
  llava-interleave-qwen-0.5b    : llava-hf/llava-interleave-qwen-0.5b-hf
  llava-interleave-qwen-7b      : llava-hf/llava-interleave-qwen-7b-hf
  llava-onevision-0.5b-ov       : llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  llava-onevision-7b-ov         : llava-hf/llava-onevision-qwen2-7b-ov-hf
  llava-onevision-72b-ov        : llava-hf/llava-onevision-qwen2-72b-ov-hf
  qwen-vl-chat                  : Qwen/Qwen-VL-Chat
  phi3-v                        : microsoft/Phi-3-vision-128k-instruct
  qwen2-vl-2b-instruct          : Qwen/Qwen2-VL-2B-Instruct
  qwen2-vl-7b-instruct          : Qwen/Qwen2-VL-7B-Instruct
  llama-3.2-11b-vision-instruct : meta-llama/Llama-3.2-11B-Vision-Instruct
  llama-3.2-90b-vision-instruct : meta-llama/Llama-3.2-90B-Vision-Instruct
</code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="Supported models:
  Model ID                      : HuggingFace Path
  ------------------------------------------------
  llava-1.5-7b                  : llava-hf/llava-1.5-7b-hf
  llava-1.5-13b                 : llava-hf/llava-1.5-13b-hf
  llava-next-video-7b           : llava-hf/LLaVA-NeXT-Video-7B-hf
  llava-next-video-7b-32k       : llava-hf/LLaVA-NeXT-Video-7B-32K-hf
  llava-next-video-34b          : llava-hf/LLaVA-NeXT-Video-34B-hf
  llava-interleave-qwen-0.5b    : llava-hf/llava-interleave-qwen-0.5b-hf
  llava-interleave-qwen-7b      : llava-hf/llava-interleave-qwen-7b-hf
  llava-onevision-0.5b-ov       : llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  llava-onevision-7b-ov         : llava-hf/llava-onevision-qwen2-7b-ov-hf
  llava-onevision-72b-ov        : llava-hf/llava-onevision-qwen2-72b-ov-hf
  qwen-vl-chat                  : Qwen/Qwen-VL-Chat
  phi3-v                        : microsoft/Phi-3-vision-128k-instruct
  qwen2-vl-2b-instruct          : Qwen/Qwen2-VL-2B-Instruct
  qwen2-vl-7b-instruct          : Qwen/Qwen2-VL-7B-Instruct
  llama-3.2-11b-vision-instruct : meta-llama/Llama-3.2-11B-Vision-Instruct
  llama-3.2-90b-vision-instruct : meta-llama/Llama-3.2-90B-Vision-Instruct" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: 500; line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: rgb(9, 105, 218); background-color: rgba(0, 0, 0, 0); box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1), background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: 8px !important; width: 28px; height: 28px;"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"></svg></clipboard-copy></div></div><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">✋ 没有看到你想要的模型？查看本</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/add_new_model.md" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">指南</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">，了解如何添加新模型的分步说明。</font></font></p></details>

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: 16px;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: 600;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">1. 准备微调数据</font></font></b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">与 LLaVA 类似，我们希望数据位于包含字典列表的 json 文件中，其中每个字典都是一个样本。</font></font></p><div class="highlight highlight-source-json notranslate position-relative overflow-auto" dir="auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; margin-bottom: 16px; display: flex; justify-content: space-between; background-color: rgb(246, 248, 250);"><pre style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: 16px; overflow: auto; line-height: 1.45; color: rgb(31, 35, 40); background-color: rgb(246, 248, 250); border-radius: 6px; word-break: normal; min-height: 52px;">[
    {
        <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"system_prompt"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>You are a helpful assistant.<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>,
        <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"video"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>path/to/video1.mp4<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>,
        <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"conversations"</span>: [
            {
                <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"from"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>human<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>,
                <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"value"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>&lt;video&gt;What is this video about?<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>
            },
            {
                <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"from"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>gpt<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>,
                <span class="pl-ent" style="box-sizing: border-box; color: rgb(5, 80, 174);">"value"</span>: <span class="pl-s" style="box-sizing: border-box; color: rgb(10, 48, 105);"><span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span>This video shows a baby crying.<span class="pl-pds" style="box-sizing: border-box; color: rgb(10, 48, 105);">"</span></span>
            },
        ]
    }
]</pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="[
    {
        &quot;system_prompt&quot;: &quot;You are a helpful assistant.&quot;,
        &quot;video&quot;: &quot;path/to/video1.mp4&quot;,
        &quot;conversations&quot;: [
            {
                &quot;from&quot;: &quot;human&quot;,
                &quot;value&quot;: &quot;<video>What is this video about?&quot;
            },
            {
                &quot;from&quot;: &quot;gpt&quot;,
                &quot;value&quot;: &quot;This video shows a baby crying.&quot;
            },
        ]
    }
]" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: 500; line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: rgb(9, 105, 218); background-color: rgba(0, 0, 0, 0); box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1), background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: 8px !important; width: 28px; height: 28px;"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"></svg></clipboard-copy></div></div><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">图像和视频标记假定为</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">&lt;image&gt;</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">和</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">&lt;video&gt;</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">。我们采用这种格式是为了提高可读性。我们的数据集实现足够通用，可以支持此格式的变体，例如，一个样本中包含多个图像/视频输入、纯文本样本等。更多详情，请参阅</font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_data" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;">数据</font></a></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/dataset.md" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">集文档</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">，了解此 JSON 文件的灵活性。example_data 下还有多个示例 JSON 文件</font><font style="box-sizing: border-box; vertical-align: inherit;">可供参考。</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_data" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"></font></a><font style="box-sizing: border-box; vertical-align: inherit;"></font></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">除了这个 json 文件之外，实际的视频和图片默认存储在相应的文件夹中，因此 json 文件中的路径应该是相对于视频/图片根文件夹的路径。或者，路径也可以是绝对路径。</font></font></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><g-emoji class="g-emoji" alias="warning" style="box-sizing: border-box; display: inline-block; min-width: 1ch; font-family: &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;; font-size: 1.25em; font-weight: 400; line-height: 1; vertical-align: -0.075em; font-style: normal !important;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">⚠️</font></font></g-emoji><span>&nbsp;</span><strong style="box-sizing: border-box; font-weight: 600;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">如果您的训练数据集中包含纯文本条目：</font></font></strong><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">如果 1) 您的条目数为 1，或 2) 纯文本实例的数量超过多模态实例的数量，则训练可能会在某个时刻失败</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">per_device_batch_size</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">。这是由于 Deepspeed 的限制/错误造成的。如果以上两个条件都不满足，不用担心，我们会为您提供帮助。</font></font></p></details>

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: 16px;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: 600;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">2. 进行微调</font></font></b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">修改示例训练 bash 脚本</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_scripts/example_video.sh" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">example_video.sh</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">或</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_image.sh" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">example_image.sh</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">（除了模型 ID 和数据集文件路径不同外，两者没有区别），以指定包括目标模型、数据路径等参数。脚本中提供了注释，解释了每个参数的含义。然后，只需运行 bash 脚本</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">bash example_scripts/example_video.sh</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">或即可开始训练</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">bash example_scripts/example_image.sh</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">。请注意，要准确运行提供的</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_scripts/example_video.sh" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">example_video.sh</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">脚本，您需要从 ShareGPT4Video 下载视频片段；请参阅</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/example_data/videos/ego4d/README.md" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">此处</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">的说明。</font></font></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">📈</font></font><em style="box-sizing: border-box;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">如果您更喜欢图形界面</font></font></em><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">，只需运行</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">python webui.py</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">即可启动 gradio 界面进行微调。</font></font></p></details>

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: 16px;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: 600;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">3. 使用微调模型进行推理</font></font></b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 16px;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">这里的关键是正确加载经过微调的模型，之后的一切操作都与使用 Huggingface 的相应模型进行推理相同。请参阅</font></font><a href="https://github.com/JackyLiuAI/lmms-ft/blob/main/docs/inference.md" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">推理文档</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">了解更多详细信息，包括如何</font></font><code style="box-sizing: border-box; font-family: &quot;Monaspace Neon&quot;, ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, &quot;Liberation Mono&quot;, monospace; font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: rgba(129, 139, 152, 0.12); border-radius: 6px;">merge_lora_weights.py</code><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">轻松获取独立模型。您也可以参考</font></font><a href="https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing" rel="nofollow" style="box-sizing: border-box; background-color: rgba(0, 0, 0, 0); color: rgb(9, 105, 218); text-decoration: underline; text-underline-offset: 0.2rem;"><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">此 Colab</font></font></a><font style="box-sizing: border-box; vertical-align: inherit;"><font style="box-sizing: border-box; vertical-align: inherit;">获取完整示例。</font></font></p></details>

## 致谢



我们要感谢 huggingface 团队积极地将最新模型集成到 Transformers 库中。此外，HF 员工[Niels Rogge](https://github.com/NielsRogge)和[Raushan Turganbay编写的示例微调脚本（例如](https://github.com/zucchini-nlp)[这个](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb)、[这个](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa-NeXT/Fine_tune_LLaVaNeXT_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb)和[这个](https://colab.research.google.com/drive/1dTdro-k7NFqRgGq5-TlGHM-6k2sYQhXp#scrollTo=4ccbd183-f15a-4f94-a526-9ceeec3f61e0)）非常有用，为这个代码库奠定了基础。我们还要特别感谢[Raushan Turganbay](https://github.com/zucchini-nlp)对此项目的慷慨讨论和反馈。

该代码库借鉴、启发或建立在以下代码、存储库和/或库之上：[LLaVA](https://github.com/haotian-liu/LLaVA)、[Qwen](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py)、[transformers](https://github.com/huggingface/transformers)等。

## 引文

如果您在研究/项目中使用 lmms-finetune，我们将非常高兴您可以 1）给我们一个星星，2）与其他人分享这个 repo，或 3）引用这个代码库：

```
@software{Zhang_lmms-finetune,
author = {Zhang, Jingyang and Lin, Yueqian},
license = {Apache-2.0},
title = {{lmms-finetune}},
url = {https://github.com/zjysteven/lmms-finetune}
}
```

