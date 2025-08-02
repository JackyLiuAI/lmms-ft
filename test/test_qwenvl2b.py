import os
import json
import torch
from typing import Dict, List
from tqdm import tqdm
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class QWen2VLInfer:
    '''
    使用QWen2VL模型进行支持多组图文示例的多卡推理模型
        message: 参考示例构造
        min_pixels (int, optional)
        max_pixels (int, optional)
        max_new_tokens (int, optional)
    '''
    def __init__(self,
                 message: list[dict] = [{}],
                 min_pixels=256*28*28,
                 max_pixels=512*28*28,
                 max_new_tokens=128,
                 ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.output_text = ''
        self.max_new_tokens = max_new_tokens
        self.message = message
        
    def update(self, **kwargs):
        """
        以**kwargs方式更新推理时的参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @torch.inference_mode()
    def initialize(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        '''
        用于加载模型权重，需要在初始化后运行
        '''
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "/data/ljq/lmms-finetune/checkpoints/cri_qwen2-vl-2b-instruct_lora-True_qlora-False",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    @torch.inference_mode()
    def infer(self):
        '''
        运行initialize之后进行推理，如变更参数需要调用update方法
        '''
        messages = self.message
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        self.output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return self.output_text


class CrisisMmdTester:
    def __init__(self, test_data_path: str, output_path: str = "qwen2vl_test_results.json"):
        self.test_data_path = test_data_path
        self.output_path = output_path
        self.results = []
        
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # 定义类别映射
        self.category_mapping = {
            "Infrastructure and utility damage": 0,
            "Vehicle damage": 1,
            "Rescue, volunteering, or donation effort": 2,
            "Injured or dead people": 3,
            "Affected individuals": 4,
            "Other relevant information": 5,
            "Not relevant or can't judge": 6,
            "Missing or found people": 7
        }
        
        # 反向映射
        self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
        
    def construct_message(self, item: Dict, image_base_path: str = "/data/ljq/lmms-finetune/SD_datasets/crisismmd/"):
        """构造消息，注意<image>的嵌入位置"""
        human_message = item["conversations"][0]["value"]
        image_path = os.path.join(image_base_path, item["image"])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
            
        # 从human_message中提取文本部分（去掉<image>标签）
        text_content = human_message.replace("<image>", "").strip()
        
        # 构造消息格式，按照QWen2VL的要求
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content},
                    {"type": "image", "image": image_path}
                ]
            }
        ]
        
        return message
    
    def extract_category(self, response: str) -> int:
        """从模型响应中提取类别"""
        response_lower = response.lower().strip()
        
        # 直接匹配类别名称
        for category, idx in self.category_mapping.items():
            if category.lower() in response_lower:
                return idx
                
        # 匹配数字
        for i in range(8):
            if str(i) in response_lower:
                return i
                
        # 默认返回6（无法判断）
        return 6
    
    def calculate_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict:
        """计算评估指标"""
        # 计算准确率
        accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
        
        # 计算每个类别的精确率、召回率、F1
        unique_labels = list(set(true_labels + pred_labels))
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for label in unique_labels:
            # 计算该类别的TP, FP, FN
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
            
            # 计算精确率、召回率、F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # 计算宏平均和加权平均
        macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        # 加权平均（按类别频率加权）
        label_counts = {label: true_labels.count(label) for label in unique_labels}
        total_count = len(true_labels)
        
        weighted_precision = sum(precision_scores[i] * label_counts[unique_labels[i]] for i in range(len(unique_labels))) / total_count
        weighted_recall = sum(recall_scores[i] * label_counts[unique_labels[i]] for i in range(len(unique_labels))) / total_count
        weighted_f1 = sum(f1_scores[i] * label_counts[unique_labels[i]] for i in range(len(unique_labels))) / total_count
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1
        }
    
    def test_qwen2vl(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", sample_size: int = None):
        """测试Qwen2-VL模型"""
        print(f"\nTesting Qwen2-VL model: {model_id}")
        
        try:
            # 初始化模型
            model_infer = QWen2VLInfer()
            model_infer.initialize(model_id=model_id)
            print(f"Model loaded successfully.")
            
            # 准备测试数据
            test_samples = self.test_data[:sample_size] if sample_size else self.test_data
            
            model_results = []
            
            for i, item in enumerate(tqdm(test_samples, desc=f"Testing {model_id}")):
                # 构造消息
                message = self.construct_message(item)
                if message is None:
                    continue
                    
                # 获取真实标签
                true_label = item["conversations"][1]["value"]
                true_category = self.category_mapping.get(true_label, 6)
                
                # 更新模型消息并进行推理
                model_infer.update(message=message)
                response = model_infer.infer()
                
                # 提取预测类别
                pred_category = self.extract_category(response)
                
                result = {
                    "sample_id": i,
                    "image": item["image"],
                    "true_label": true_label,
                    "true_category": true_category,
                    "model_response": response,
                    "pred_category": pred_category,
                    "correct": true_category == pred_category
                }
                
                model_results.append(result)
                
                # 每50个样本清理一次内存
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # 计算指标
            true_labels = [r["true_category"] for r in model_results]
            pred_labels = [r["pred_category"] for r in model_results]
            
            metrics = self.calculate_metrics(true_labels, pred_labels)
            
            result_data = {
                "model_id": model_id,
                "model_family": "qwen2-vl",
                "metrics": metrics,
                "predictions": model_results,
                "total_samples": len(model_results)
            }
            
            self.results.append(result_data)
            
            print(f"Results for {model_id}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
            print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            
            # 清理内存
            del model_infer
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing model {model_id}: {str(e)}")
            
    def save_results(self):
        """保存结果到JSON文件"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {self.output_path}")
        
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*50)
        print("QWEN2-VL TESTING SUMMARY")
        print("="*50)
        
        for result in self.results:
            model_id = result["model_id"]
            metrics = result["metrics"]
            print(f"\n{model_id}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
            print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"  Weighted Precision: {metrics['weighted_precision']:.4f}")
            print(f"  Weighted Recall: {metrics['weighted_recall']:.4f}")
            print(f"  Total samples: {result['total_samples']}")


def main():
    # 配置路径
    test_data_path = "/data/ljq/lmms-finetune/SD_datasets/crisismmd/test.json"
    output_path = "/data/ljq/lmms-finetune/qwen2vl_test_results.json"
    
    # 创建测试器
    tester = CrisisMmdTester(test_data_path, output_path)
    
    # 测试Qwen2-VL-2B-Instruct模型 - 测试所有样本
    tester.test_qwen2vl(
        model_id="Qwen/Qwen2-VL-2B-Instruct"
        # 移除 sample_size 参数，测试所有样本
    )
    
    # 保存结果
    tester.save_results()
    
    # 打印总结
    tester.print_summary()


if __name__ == "__main__":
    main()