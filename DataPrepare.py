import os
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, Features, Value # 确保导入了 Dataset 和 Features

# 设置 Hugging Face 环境变量，优先使用镜像站下载（如果需要下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class DatasetPreparer:
    """
    负责加载、筛选和格式化用于模型微调的数据集。
    """
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        初始化数据集准备器。

        Args:
            tokenizer (AutoTokenizer): 用于分词的 tokenizer 实例。
            max_length (int): 文本序列的最大长度。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _format_instruction_dataset(self, example):
        """
        将指令微调数据集的单个样本格式化为模型期望的聊天对话格式。
        """
        messages = []
        # 根据你提供的图片，使用 'instruct' 列作为指令，'input' 和 'output' 保持不变
        if example.get("instruct"):
            user_content = example["instruct"]
            if example.get("input"):
                user_content += "\n" + example["input"]
            messages.append({"role": "user", "content": user_content})
        
        if example.get("output"):
            messages.append({"role": "assistant", "content": example["output"]})
        
        # 使用 tokenizer 的 chat template 将对话历史转化为模型输入文本
        # add_generation_prompt=False 因为这里是训练数据，output 是标签
        return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    def _tokenize_function(self, examples):
        """
        对格式化后的文本进行分词。
        """
        return self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)

    def prepare_for_finetuning(self, dataset_name: str, lang_filter: str = None):
        """
        加载、筛选并准备数据集以用于微调。

        Args:
            dataset_name (str): Hugging Face 数据集名称，例如 "fm-universe/FM-alpaca"。
            lang_filter (str, optional): 用于筛选 'lang' 列的字符串值，例如 'ACSL'。
                                        如果为 None，则不进行语言筛选。

        Returns:
            tuple: (train_dataset, data_collator)
                train_dataset (Dataset): 格式化并分词后的训练数据集。
                data_collator (DataCollatorForLanguageModeling): 数据整理器。
        """
        print(f"正在加载数据集: {dataset_name}...")
        raw_dataset = load_dataset(dataset_name)
        if 'train' not in raw_dataset:
            raise ValueError(f"数据集 '{dataset_name}' 不包含 'train' 分割。请检查数据集结构。")
        dataset = raw_dataset['train']
        print(f"数据集 '{dataset_name}' 加载完成，包含 {len(dataset)} 条样本。")

        if lang_filter:
            print(f"正在筛选 'lang' 为 '{lang_filter}' 的数据...")
            dataset = dataset.filter(lambda example: example.get("lang") == lang_filter)
            print(f"筛选完成，剩余 {len(dataset)} 条样本。")

        print("正在格式化和分词数据集...")
        # 应用格式化函数，然后进行分词
        # remove_columns 确保在 map 之后移除原始列，只留下 'text' 列
        # formatted_dataset.column_names 是原始数据集的列名，需要传递给 remove_columns
        formatted_dataset = dataset.map(self._format_instruction_dataset, remove_columns=dataset.column_names)
        
        # 确保分词后的数据集移除了原始的 'text' 列，因为 tokenized_dataset 会包含 'input_ids', 'attention_mask' 等
        tokenized_dataset = formatted_dataset.map(self._tokenize_function, batched=True, remove_columns=["text"])
        print("数据集格式化和分词完成。")

        # 定义 Data Collator
        # DataCollatorForLanguageModeling 需要一个 tokenizer 来进行动态填充
        # 这里为了单独测试，我们并不真正需要 data_collator 的功能，但为了接口一致性仍然返回
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        print("数据整理器已定义。")

        return tokenized_dataset, data_collator

# 示例用法
if __name__ == "__main__":
    # 1. 模拟加载一个 tokenizer (无需加载整个模型权重)
    # 使用 Qwen1.5-0.5B 的 tokenizer，因为它是轻量级的且具有聊天模板
    TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
    print(f"正在加载测试 tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("测试 tokenizer 加载完成。")

    # 2. 定义数据集名称和筛选条件
    DATASET_NAME = "fm-universe/FM-alpaca"
    LANG_FILTER = "TLA" # 你希望筛选的语言

    # 3. 初始化 DatasetPreparer
    dataset_preparer = DatasetPreparer(tokenizer=tokenizer, max_length=512)

    # 4. 调用 prepare_for_finetuning 方法
    print(f"\n开始准备数据集 '{DATASET_NAME}'，并筛选 'lang' 为 '{LANG_FILTER}' 的数据...")
    train_dataset, data_collator = dataset_preparer.prepare_for_finetuning(
        dataset_name=DATASET_NAME,
        lang_filter=LANG_FILTER
    )

    # 5. 打印处理结果
    print("\n--- 数据处理结果 ---")
    print(f"处理后的训练数据集包含 {len(train_dataset)} 条样本。")

    if len(train_dataset) > 0:
        print("\n第一条样本的输入ID (tokenized text):")
        # 打印 input_ids 的前几个，以及解码后的文本
        print(train_dataset[0]["input_ids"][:50]) # 打印前50个token ID
        print("\n第一条样本的解码文本 (原始格式化文本):")
        print(tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True))
        
        print("\n最后一条样本的解码文本 (原始格式化文本):")
        print(tokenizer.decode(train_dataset[-1]["input_ids"], skip_special_tokens=True))
    else:
        print("没有筛选出任何样本。请检查数据集名称和筛选条件。")

    print("\n数据处理模块测试完成。")

