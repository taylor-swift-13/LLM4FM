import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, base_model_name="Qwen/Qwen3-0.6B", fine_tuned_model_path="fine_tuned_models"):
        self.base_model_name = base_model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        self.base_tokenizer = None
        self.base_model = None
        self.fine_tuned_tokenizer = None
        self.fine_tuned_model = None
        
    def load_models(self):
        """加载原模型和微调后的模型"""
        logger.info("加载原模型...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info("加载微调后的模型...")
        self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
        if self.fine_tuned_tokenizer.pad_token is None:
            self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token
        
        self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            self.fine_tuned_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info("模型加载完成")
    
    def load_test_data(self, dataset_name="fm-universe/FM-bench", lang_filter="ACSL", num_samples=3):
        """加载测试数据"""
        logger.info(f"加载测试数据集: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # 筛选ACSL数据
        if lang_filter:
            dataset = dataset.filter(lambda x: x.get('lang') == lang_filter)
        
        # 获取所有分割的数据
        all_data = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            for item in split_data:
                all_data.append(item)
        
        # 选择前num_samples个样本
        test_data = all_data[:num_samples]
        logger.info(f"选择了 {len(test_data)} 个测试样本")
        
        return test_data
    
    def format_question(self, item):
        """格式化问题"""
        question = ""
        if item.get("instruct"):
            question += item["instruct"].strip()
        if item.get("input"):
            input_text = item["input"].strip()
            if input_text:
                if question:
                    question += "\n" + input_text
                else:
                    question = input_text
        
        return question
    
    def generate_answer(self, model, tokenizer, question, max_new_tokens=2048):
        """使用模型生成答案"""
        # 格式化输入
        messages = [
            {"role": "user", "content": question}
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(formatted_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 移动到模型设备
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 生成答案
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # 解码生成的答案
        input_length = input_ids.shape[1]
        generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        
        # 移除生成的think标签
        generated_text = generated_text.replace('<think>', '').replace('</think>', '')
        
        return generated_text.strip()
    
    def compare_models(self, test_data):
        """比较两个模型的回答"""
        results = []
        
        for i, item in enumerate(test_data):
            logger.info(f"\n=== 问题 {i+1} ===")
            
            # 格式化问题
            question = self.format_question(item)
            true_answer = item.get("output", "").strip()
            
            logger.info(f"问题: {question}")
            logger.info(f"标准答案: {true_answer}")
            
            # 使用原模型生成答案
            logger.info("使用原模型生成答案...")
            base_answer = self.generate_answer(self.base_model, self.base_tokenizer, question)
            logger.info(f"原模型答案: {base_answer}")
            
            # 使用微调模型生成答案
            logger.info("使用微调模型生成答案...")
            fine_tuned_answer = self.generate_answer(self.fine_tuned_model, self.fine_tuned_tokenizer, question)
            logger.info(f"微调模型答案: {fine_tuned_answer}")
            
            # 保存结果
            result = {
                "question_id": i + 1,
                "question": question,
                "true_answer": true_answer,
                "base_model_answer": base_answer,
                "fine_tuned_model_answer": fine_tuned_answer
            }
            results.append(result)
            
            logger.info("-" * 50)
        
        return results
    
    def save_results(self, results, output_file="model_comparison_results.json"):
        """保存比较结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到 {output_file}")
    
    def print_summary(self, results):
        """打印比较摘要"""
        logger.info("\n=== 模型比较摘要 ===")
        
        for result in results:
            logger.info(f"\n问题 {result['question_id']}:")
            logger.info(f"  问题: {result['question'][:100]}...")
            logger.info(f"  标准答案: {result['true_answer'][:100]}...")
            logger.info(f"  原模型答案: {result['base_model_answer'][:100]}...")
            logger.info(f"  微调模型答案: {result['fine_tuned_model_answer'][:100]}...")

def main():
    # 创建比较器
    comparator = ModelComparator()
    
    # 加载模型
    comparator.load_models()
    
    # 加载测试数据
    test_data = comparator.load_test_data(num_samples=1)
    
    # 比较模型
    results = comparator.compare_models(test_data)
    
    # 保存结果
    comparator.save_results(results)
    
    # 打印摘要
    comparator.print_summary(results)

if __name__ == "__main__":
    main() 