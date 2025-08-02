import os
import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, concatenate_datasets

# 设置 Hugging Face 环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class DatasetPreparer:
    """
    负责加载、筛选和格式化用于模型微调的数据集。
    """
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用模型专用的角色标记
        self.system_token = "<|im_start|>"
        self.user_token = "<|im_start|>user"
        self.assistant_token = "<|im_start|>assistant"
        self.end_token = "<|im_end|>"
        
        # 确保这些特殊标记在 tokenizer 中存在
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.system_token,
                self.user_token,
                self.assistant_token,
                self.end_token
            ]
        })
        # 获取特殊标记的ID
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(self.assistant_token)
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_token)

    def _format_instruction_dataset(self, example):
        """使用模型原生格式格式化对话，优化空格"""
        messages = []
        
        # 用户消息 - 优化空格
        user_content = ""
        if example.get("instruct"):
            user_content = example["instruct"].strip()
            if example.get("input"):
                input_text = example["input"].strip()
                if input_text:
                    user_content += "\n" + input_text
        messages.append({"role": "user", "content": user_content})
        
        # 助手回复 - 优化空格
        if example.get("output"):
            output_content = example["output"].strip()
            if output_content:
                messages.append({"role": "assistant", "content": output_content})
        
        # 转换为模型原生格式
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 保留think标签，只清理多余的空格和换行
        # formatted_text = formatted_text.replace('<think>', '').replace('</think>', '')
        
        # 清理多余的空格和换行
        formatted_text = ' '.join(formatted_text.split())
        
        return {"text": formatted_text}

    def _tokenize_and_mask_function(self, examples):
        """修复版：分词并设置损失掩码，只有output部分计算loss"""
        # 分词
        tokenized = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # 创建标签张量，默认全部忽略
        labels = torch.full_like(tokenized["input_ids"], -100)
        
        # 识别助手回复位置（即output部分）
        for i in range(labels.shape[0]):
            input_ids = tokenized["input_ids"][i]
            
            # 查找助手标记位置
            assistant_indices = torch.where(input_ids == self.assistant_token_id)[0]
            
            if len(assistant_indices) > 0:
                start_idx = assistant_indices[0] + 1  # 从助手标记后开始
                
                # 查找结束位置
                end_indices = torch.where(input_ids == self.end_token_id)[0]
                end_idx = end_indices[-1] if len(end_indices) > 0 else len(input_ids)
                
                # 确保结束位置在序列长度内
                end_idx = min(end_idx, len(input_ids))
                
                # 只有助手回复部分（output）计算loss
                if start_idx < end_idx:
                    labels[i, start_idx:end_idx] = input_ids[start_idx:end_idx]
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    def prepare_for_finetuning(self, dataset_name: str, lang_filter: str = None):
        print(f"正在加载数据集: {dataset_name}...")
        # 加载整个数据集（包括所有分割）
        dataset_dict = load_dataset(dataset_name)
        print(f"数据集 '{dataset_name}' 加载完成")
        
        # 合并所有分割的数据
        all_data = []
        for split_name in dataset_dict:
            split_data = dataset_dict[split_name]
            all_data.extend(split_data)
        
        # 转换为Dataset对象
        full_dataset = Dataset.from_list(all_data)
        print(f"合并后数据集包含 {len(full_dataset)} 条样本")
        
        if lang_filter:
            print(f"正在筛选 'lang' 为 '{lang_filter}' 的数据...")
            full_dataset = full_dataset.filter(lambda example: example.get("lang") == lang_filter)
            print(f"筛选完成，剩余 {len(full_dataset)} 条样本")

        print("正在格式化和分词数据集...")
        # 获取实际列名
        actual_columns = full_dataset.column_names
        
        # 格式化对话
        formatted_dataset = full_dataset.map(
            self._format_instruction_dataset,
            remove_columns=actual_columns
        )
        
        # 分词并设置损失掩码
        tokenized_dataset = formatted_dataset.map(
            self._tokenize_and_mask_function,
            batched=True,
            batch_size=64,
            remove_columns=["text"]
        )
        print("数据集格式化和分词完成")

        # 创建数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        print("数据整理器已定义")

        return tokenized_dataset, data_collator

# 测试代码
if __name__ == "__main__":
    # 1. 加载tokenizer
    TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
    print(f"正在加载测试 tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("测试 tokenizer 加载完成")
    
    # 2. 定义数据集名称和筛选条件
    DATASET_NAME = "fm-universe/FM-alpaca"
    LANG_FILTER = "ACSL"
    
    # 3. 初始化 DatasetPreparer
    dataset_preparer = DatasetPreparer(tokenizer=tokenizer, max_length=512)
    
    # 4. 准备数据
    print(f"\n开始准备数据集 '{DATASET_NAME}'，并筛选 'lang' 为 '{LANG_FILTER}' 的数据...")
    train_dataset, data_collator = dataset_preparer.prepare_for_finetuning(
        dataset_name=DATASET_NAME,
        lang_filter=LANG_FILTER
    )
    
    # 5. 验证结果
    print("\n--- 数据处理结果 ---")
    print(f"处理后的训练数据集包含 {len(train_dataset)} 条样本")
    
    if len(train_dataset) > 0:
        print("\n第一条样本的输入ID:")
        print(train_dataset[0]["input_ids"])
        print("\n第一条样本的损失掩码 (labels):")
        print(train_dataset[0]["labels"])
        
        print("\n完整输入解码:")
        print(tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False))
        
        print("\n掩码解码 (仅显示需计算损失的部分):")
        labels = train_dataset[0]["labels"]
        # 将labels转换为张量
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        mask_positions = torch.ne(labels, -100)  # 查找非-100的位置
        mask_positions = mask_positions.cpu().numpy()  # 转换为numpy数组
        masked_tokens = np.array(train_dataset[0]["input_ids"])[mask_positions]
        print(tokenizer.decode(masked_tokens.tolist(), skip_special_tokens=True))
    else:
        print("没有筛选出任何样本")
    
    print("\n数据处理模块测试完成")