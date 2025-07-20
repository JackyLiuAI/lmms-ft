# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
# "--model_id", type=str, required=True)
# "--model_local_path", type=str, default="")
# "--model_path", type=str, required=True)
# "--model_save_path", type=str, default="")
# "--load_model", action="store_true")
# "--load_4bit", action="store_true")


python merge_lora_weights.py \
    --model_id qwen2-vl-2b-instruct \
    --model_local_path Qwen/Qwen2-VL-2B-Instruct \
    --model_path checkpoints/qwen2-vl-2b-instruct_lora-True_qlora-False \
    --model_save_path ./output \
    --load_model