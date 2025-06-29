import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from DataPrepare import DatasetPreparer

# 设置 Hugging Face 环境变量，优先使用镜像站下载（如果需要下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class FineTuner:
    """
    用于使用 PEFT (LoRA) 微调 Qwen 模型的类。
    """
    def __init__(self, model_name: str="Qwen/Qwen3-0.6B", output_dir: str = "./qwen_finetuned_results"):
        """
        初始化微调器。

        Args:
            model_name (str): Hugging Face 模型名称或本地路径，例如 "Qwen/Qwen3-0.6B"。
            output_dir (str): 保存模型检查点和最终 LoRA 适配器的目录。
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.data_collator = None
        self.trainer = None
        self.dataset_preparer = None # 新增数据集准备器实例

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        加载分词器和基础模型。
        """
        print(f"正在加载分词器: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print(f"正在加载基础模型: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        print("基础模型加载完成。")

    def prepare_dataset(self, dataset_name: str, lang_filter: str = None):
        """
        通过 DatasetPreparer 准备用于微调的数据集。

        Args:
            dataset_name (str): Hugging Face 数据集名称。
            lang_filter (str, optional): 用于筛选 'lang' 列的字符串值。
        """
        # 实例化 DatasetPreparer
        self.dataset_preparer = DatasetPreparer(tokenizer=self.tokenizer)
        # 调用 DatasetPreparer 的 prepare_for_finetuning 方法
        self.train_dataset, self.data_collator = self.dataset_preparer.prepare_for_finetuning(
            dataset_name=dataset_name,
            lang_filter=lang_filter
        )


    def configure_peft(self, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, target_modules: list = None):
        """
        配置并应用 PEFT (LoRA)。

        Args:
            r (int): LoRA 的秩。
            lora_alpha (int): LoRA 的缩放因子。
            lora_dropout (float): LoRA 层上的 Dropout 概率。
            target_modules (list): 指定要应用 LoRA 的模型层。
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"] # Qwen 模型的默认目标模块

        print("正在配置 PEFT (LoRA)...")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        print("LoRA 适配器已应用。可训练参数如下：")
        self.model.print_trainable_parameters()

    def train(self, training_args: TrainingArguments):
        """
        开始模型的训练。

        Args:
            training_args (TrainingArguments): Hugging Face TrainingArguments 对象。
        """
        if self.train_dataset is None or self.data_collator is None:
            raise ValueError("请先调用 prepare_dataset() 方法准备训练数据。")

        print("正在初始化 Trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )

        print("开始训练...")
        self.trainer.train()
        print("训练完成。")

    def save_fine_tuned_model(self, save_path: str = None):
        """
        保存微调后的 PEFT 模型（LoRA 适配器）和分词器。

        Args:
            save_path (str, optional): 保存模型的路径。默认为初始化时定义的 output_dir/final_adapters。
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "final_adapters")
        os.makedirs(save_path, exist_ok=True)

        print(f"正在保存微调后的 PEFT 适配器到: {save_path}")
        self.trainer.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("保存完成。")

    @classmethod
    def load_fine_tuned_model(cls, model_name: str, peft_model_path: str, device: str = None):
        """
        类方法：加载带有 LoRA 适配器的微调模型。

        Args:
            model_name (str): 原始基础模型的名称或路径。
            peft_model_path (str): 保存 LoRA 适配器的路径。
            device (str, optional): 加载模型到的设备（"cuda" 或 "cpu"）。默认为自动检测。

        Returns:
            peft.PeftModel: 加载并应用了 LoRA 适配器的模型。
            transformers.AutoTokenizer: 对应的分词器。
        """


        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载基础模型: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"正在加载 PEFT 适配器: {peft_model_path}")
        fine_tuned_model = PeftModel.from_pretrained(base_model, peft_model_path)
        print("模型加载并应用 LoRA 适配器完成。")

        return fine_tuned_model, base_tokenizer


# 示例用法
if __name__ == "__main__":
    # 1. 定义模型名称和输出目录
    QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"
    # 如果你本地的模型文件夹真的是 Qwen/Qwen3-0.6B，请改为 "Qwen/Qwen3-0.6B"
    FINE_TUNED_OUTPUT_DIR = "./qwen3_alpaca_acsl_finetuned_results_"

    # 2. 指定要使用的 Hugging Face 数据集名称和筛选条件
    DATASET_NAME = "fm-universe/FM-alpaca"
    LANG_FILTER = "ACSL" # 筛选 'lang' 列为 'ACSL' 的数据

    # 3. 初始化微调器
    fine_tuner = FineTuner(model_name=QWEN_MODEL_NAME, output_dir=FINE_TUNED_OUTPUT_DIR)

    # 4. 准备数据集 (现在通过 prepare_dataset 方法内部调用 DatasetPreparer)
    fine_tuner.prepare_dataset(DATASET_NAME, lang_filter=LANG_FILTER)

    # 5. 配置 PEFT (LoRA)
    fine_tuner.configure_peft(r=8, lora_alpha=16, lora_dropout=0.1)

    # 6. 定义训练参数
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {device_name}")

    training_args = TrainingArguments(
        label_names=["labels"],  
        output_dir=FINE_TUNED_OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir=os.path.join(FINE_TUNED_OUTPUT_DIR, "logs"),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        max_grad_norm=0.3,
        report_to="none",
        push_to_hub=False,
    )

    # 7. 开始训练
    fine_tuner.train(training_args)

    # 8. 保存微调后的 LoRA 适配器
    fine_tuner.save_fine_tuned_model()

    print("\n--- 微调流程结束 ---")

    # --- 加载并测试微调后的模型 ---
    print("\n--- 正在加载并测试微调后的模型 ---")
    peft_adapters_path = os.path.join(FINE_TUNED_OUTPUT_DIR, "final_adapters")

    loaded_model, loaded_tokenizer = FineTuner.load_fine_tuned_model(
        model_name=QWEN_MODEL_NAME,
        peft_model_path=peft_adapters_path,
        device=device_name
    )

    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "请用一句话概括 PEFT 微调的核心思想。"}
    ]
    
    text = loaded_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = loaded_tokenizer(text, return_tensors="pt").to(loaded_model.device)

    print("\n正在生成推理结果...")
    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=loaded_tokenizer.eos_token_id,
            pad_token_id=loaded_tokenizer.pad_token_id
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    generated_text = loaded_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("\n生成结果:")
    print(generated_text)



