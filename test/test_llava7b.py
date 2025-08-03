import os
import json
import torch
from typing import Dict, List
from tqdm import tqdm
import gc
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import ast

class LLaVAInfer:
    '''
    使用LLaVA模型进行支持多组图文示例的多卡推理模型
        message: 参考示例构造
        max_new_tokens (int, optional)
    '''
    def __init__(self,
                 message: list[dict] = [{}],
                 max_new_tokens=1024,
                 ):
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
    def initialize(self, model_id="llava-hf/llava-1.5-7b-hf", model_path=None):
        '''
        用于加载模型权重，需要在初始化后运行
        '''
        # 如果没有提供model_path，使用默认路径
        if model_path is None:
            model_path = "/DATA/home/ljq/Projects/lmms-ft/checkpoints/cri_llava-1.5-7b_lora-True_qlora-False"
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def infer(self):
        '''
        运行initialize之后进行推理，如变更参数需要调用update方法
        '''
        # message现在应该是construct_message返回的格式
        if not self.message or "conversation" not in self.message or "image_path" not in self.message:
            return "Error: Invalid message format"
        
        conversation = self.message["conversation"]
        image_path = self.message["image_path"]
        
        # 加载图像
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return "Error: Failed to load image"
        
        # 应用chat template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # 处理输入
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
        
        # 生成输出
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        
        # 解码输出
        raw_response = self.processor.decode(output[0][2:], skip_special_tokens=True)
        
        # 处理LLaVA的输出格式：以ASSISTANT:切分，保留后者并去掉前后空格
        if "ASSISTANT:" in raw_response:
            # 以ASSISTANT:切分，取后面的部分
            assistant_response = raw_response.split("ASSISTANT:", 1)[1]
            # 去掉前后空格
            self.output_text = assistant_response.strip()
        else:
            # 如果没有ASSISTANT:标记，直接使用原始响应
            self.output_text = raw_response.strip()
        
        return self.output_text


class CrisisMMDSDTester:
    def __init__(self, test_json_path: str, test_txt_path: str, output_path: str = "llava7b_sd_test_results.json"):
        self.test_json_path = test_json_path
        self.test_txt_path = test_txt_path
        self.output_path = output_path
        self.results = []
        
        # 加载测试数据
        with open(test_json_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # 加载SD标签信息
        self.sd_labels = self.load_sd_labels()
        
        # 定义类别映射
        self.category_mapping = {
            "Infrastructure and utility damage": 0,
            "Vehicle damage": 1,
            "Rescue, volunteering, or donation effort": 2,
            "Injured or dead people": 3,
            "Affected individuals": 4,
            "Missing or found people": 5,
            "Other relevant information": 6,
            "Not relevant or can't judge": 6
        }
        
    def load_sd_labels(self) -> Dict[str, int]:
        """从test.txt文件加载SD标签信息"""
        sd_labels = {}
        with open(self.test_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # 解析每行的字典数据
                        data = ast.literal_eval(line)
                        image_id = data['id']
                        information_label = data['information_label']
                        sd_labels[image_id] = information_label
                    except Exception as e:
                        print(f"Error parsing line: {line}, Error: {e}")
                        continue
        return sd_labels
    
    def construct_message(self, item: Dict, image_base_path: str = "/DATA/home/ljq/Projects/lmms-ft/SD_datasets/crisismmd/"):
        """构造模型输入消息"""
        image_path = os.path.join(image_base_path, item["image"])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
            
        # 提取human的问题文本
        human_text = item["conversations"][0]["value"]
        
        # 解析文本，找到<image>的位置并拆分
        parts = human_text.split('<image>')
        
        # 构造content列表
        content = []
        
        # 添加第一部分文本（如果存在）
        if parts[0].strip():
            content.append({"type": "text", "text": parts[0].strip()})
        
        # 添加图像
        content.append({"type": "image"})
        
        # 添加图像后的文本（如果存在）
        if len(parts) > 1 and parts[1].strip():
            content.append({"type": "text", "text": parts[1].strip()})
        
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "image_path": image_path
        }
    
    def extract_category(self, response: str) -> int:
        """从模型响应中提取类别"""
        response = response.strip().lower()
        
        # 直接匹配类别名称
        for category, idx in self.category_mapping.items():
            if category.lower() in response:
                return idx
        
        # 如果没有匹配到，返回默认类别
        return 6
    
    def calculate_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict:
        """计算评估指标"""
        if len(true_labels) != len(pred_labels):
            raise ValueError("True labels and predicted labels must have the same length")
        
        if len(true_labels) == 0:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0
            }
        
        # 计算准确率
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        accuracy = correct / len(true_labels)
        
        # 获取所有类别
        all_labels = sorted(set(true_labels + pred_labels))
        
        # 计算每个类别的精确率、召回率和F1
        precisions = []
        recalls = []
        f1s = []
        supports = []
        
        for label in all_labels:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = sum(1 for t in true_labels if t == label)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)
        
        # 计算宏平均
        macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        
        # 计算加权平均
        total_support = sum(supports)
        if total_support > 0:
            weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
            weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1
        }
    
    def test_and_analyze_by_sd(self, model_id: str = "llava-hf/llava-1.5-7b-hf", model_path: str = None, image_base_path: str = None):
        """测试整个数据集一次，然后按SD属性分析结果"""
        print(f"\nTesting LLaVA model: {model_id}")
        
        # 设置默认的image_base_path
        if image_base_path is None:
            image_base_path = "/DATA/home/ljq/Projects/lmms-ft/SD_datasets/crisismmd/"
        
        try:
            # 初始化模型
            model_infer = LLaVAInfer()
            model_infer.initialize(model_id=model_id, model_path=model_path)
            print(f"Model loaded successfully.")
            
            # 筛选有SD标签的样本
            valid_samples = []
            for item in self.test_data:
                image_id = item["image"]
                if image_id in self.sd_labels:
                    valid_samples.append(item)
            
            print(f"Total valid samples with SD labels: {len(valid_samples)}")
            
            # 测试所有样本
            all_results = []
            
            for i, item in enumerate(tqdm(valid_samples, desc="Testing all samples")):
                # 构造消息 - 传递image_base_path参数
                message = self.construct_message(item, image_base_path)
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
                
                # 获取SD标签
                information_label = self.sd_labels.get(item["image"], -1)
                
                result = {
                    "sample_id": i,
                    "image": item["image"],
                    "true_label": true_label,
                    "true_category": true_category,
                    "model_response": response,
                    "pred_category": pred_category,
                    "correct": true_category == pred_category,
                    "information_label": information_label
                }
                
                all_results.append(result)
                
                # 每50个样本清理一次内存
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # 清理模型内存
            del model_infer
            torch.cuda.empty_cache()
            gc.collect()
            
            # 按SD属性分析结果
            print("\nAnalyzing results by SD category...")
            
            # 分类结果
            entire_results = all_results
            non_sd_results = [r for r in all_results if r["information_label"] == 1]
            sd_results = [r for r in all_results if r["information_label"] == 0]
            
            print(f"Entire Dataset: {len(entire_results)} samples")
            print(f"Non-SD Subset (information_label=1): {len(non_sd_results)} samples")
            print(f"SD Subset (information_label=0): {len(sd_results)} samples")
            
            # 计算各个子集的指标
            datasets = {
                "Entire Dataset": entire_results,
                "Non-SD Subset": non_sd_results,
                "SD Subset": sd_results
            }
            
            results = {}
            
            for dataset_name, dataset_results in datasets.items():
                if len(dataset_results) > 0:
                    true_labels = [r["true_category"] for r in dataset_results]
                    pred_labels = [r["pred_category"] for r in dataset_results]
                    
                    metrics = self.calculate_metrics(true_labels, pred_labels)
                    
                    results[dataset_name] = {
                        "metrics": metrics,
                        "predictions": dataset_results,
                        "total_samples": len(dataset_results)
                    }
                else:
                    results[dataset_name] = {
                        "metrics": {
                            "accuracy": 0.0,
                            "macro_precision": 0.0,
                            "macro_recall": 0.0,
                            "macro_f1": 0.0,
                            "weighted_precision": 0.0,
                            "weighted_recall": 0.0,
                            "weighted_f1": 0.0
                        },
                        "predictions": [],
                        "total_samples": 0
                    }
            
            # 保存结果
            result_data = {
                "model_id": model_id,
                "model_family": "llava",
                "results_by_category": results
            }
            
            self.results.append(result_data)
            
            # 打印表格格式结果
            self.print_table_results(results)
            
        except Exception as e:
            print(f"Error testing model {model_id}: {str(e)}")
    
    def print_table_results(self, results: Dict):
        """按照表格格式打印结果"""
        print("\n" + "="*80)
        print("CrisisMMD Test Results")
        print("="*80)
        print(f"{'Dataset':<20} {'Acc(%)':<10} {'F1(%)':<10}")
        print("-"*40)
        
        for dataset_name, data in results.items():
            if data["total_samples"] > 0:
                acc = data["metrics"]["accuracy"] * 100
                f1 = data["metrics"]["macro_f1"] * 100
                print(f"{dataset_name:<20} {acc:<10.4f} {f1:<10.4f}")
            else:
                print(f"{dataset_name:<20} {'N/A':<10} {'N/A':<10}")
    
    def save_results(self):
        """保存结果到JSON文件"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {self.output_path}")


def main():
    # 配置路径
    test_json_path = "/DATA/home/ljq/Projects/lmms-ft/SD_datasets/crisismmd/test.json"
    test_txt_path = "/DATA/home/ljq/Projects/lmms-ft/SD_datasets/crisismmd/test.txt"
    output_path = "/DATA/home/ljq/Projects/lmms-ft/llava7b_sd_test_results.json"
    image_base_path = "/DATA/home/ljq/Projects/lmms-ft/SD_datasets/crisismmd/"
    
    # 设置模型路径
    model_id = "llava-hf/llava-1.5-7b-hf"
    # model_path = "llava-hf/llava-1.5-7b-hf"
    model_path = "/DATA/home/ljq/Projects/lmms-ft/checkpoints/cri_llava-1.5-7b_lora-True_qlora-False"
    
    # 创建测试器
    tester = CrisisMMDSDTester(test_json_path, test_txt_path, output_path)
    
    # 测试整个数据集并按SD属性分析
    tester.test_and_analyze_by_sd(
        model_id=model_id,
        model_path=model_path,
        image_base_path=image_base_path
    )
    
    # 保存结果
    tester.save_results()


if __name__ == "__main__":
    main()