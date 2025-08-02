import os
import json
import torch
import logging
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 导入BLEU和ROUGE相关库
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    BLEU_ROUGE_AVAILABLE = False
    logging.warning("BLEU和ROUGE库未安装，将跳过这些指标")

from data_prepare import DatasetPreparer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    """
    统一的微调类，支持带评估和不带评估两种模式
    """
    def __init__(self, 
                 model_name="Qwen/Qwen3-0.6B", 
                 train_dataset_name="fm-universe/FM-alpaca", 
                 test_dataset_name="fm-universe/FM-bench",
                 output_dir="./qwen_finetuned_results",
                 enable_evaluation=False):
        """
        初始化微调器
        
        Args:
            model_name (str): 模型名称
            train_dataset_name (str): 训练数据集名称
            test_dataset_name (str): 测试数据集名称
            output_dir (str): 输出目录
            enable_evaluation (bool): 是否启用评估模式
        """
        self.model_name = model_name
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.output_dir = output_dir
        self.enable_evaluation = enable_evaluation
        
        # 数据集相关
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.data_collator = None
        self.dataset_preparer = None
        
        # 模型相关
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # 评估结果
        self.validation_results = {}
        
        # 语言筛选
        self.LANG = "ACSL"
        
        # 设备检测
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"正在加载模型: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 确保模型参数可以计算梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("模型和分词器加载完成")

    def configure_peft(self, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=None):
        """配置PEFT (LoRA)"""
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj",       # MLP层
            ]
        
        logger.info("正在配置 PEFT (LoRA)...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 确保PEFT模型参数可以计算梯度
        for name, param in self.model.named_parameters():
            if "lora" in name or "adapter" in name:
                param.requires_grad = True
        
        logger.info("LoRA 适配器已应用")
        self.model.print_trainable_parameters()

    def prepare_datasets(self):
        """准备数据集"""
        self.dataset_preparer = DatasetPreparer(tokenizer=self.tokenizer)
        
        # 加载训练数据集
        logger.info(f"加载训练数据集: {self.train_dataset_name}")
        try:
            self.train_dataset, self.data_collator = self.dataset_preparer.prepare_for_finetuning(
                self.train_dataset_name, self.LANG
            )
            
            logger.info(f"训练数据集加载成功: {len(self.train_dataset)} 样本")
        except Exception as e:
            logger.error(f"训练数据集加载失败: {e}")
            return False
        
        # 如果启用评估，加载测试数据集并分割
        if self.enable_evaluation:
            logger.info(f"加载测试数据集: {self.test_dataset_name}")
            try:
                test_dataset, _ = self.dataset_preparer.prepare_for_finetuning(
                    self.test_dataset_name, self.LANG
                )
                logger.info(f"测试数据集加载成功: {len(test_dataset)} 样本")
                
                # 分割数据集
                self._split_test_dataset(test_dataset)
                
            except Exception as e:
                logger.error(f"测试数据集加载失败: {e}")
                self.valid_dataset = None
                self.test_dataset = None
        
        return True

    def _split_test_dataset(self, test_dataset):
        """分割测试数据集为验证集和测试集"""
        if test_dataset:
            total_size = len(test_dataset)
            validation_ratio = 0.5
            validation_size = int(total_size * validation_ratio)
            
            # 随机打乱数据集
            indices = list(range(total_size))
            random.shuffle(indices)
            
            # 创建验证集和测试集
            self.valid_dataset = test_dataset.select(indices[:validation_size])
            self.test_dataset = test_dataset.select(indices[validation_size:])
            
            logger.info(f"数据集分割完成 (验证集比例: {validation_ratio*100}%):")
            logger.info(f"  训练集: {len(self.train_dataset)} 样本")
            logger.info(f"  验证集: {len(self.valid_dataset)} 样本")
            logger.info(f"  测试集: {len(self.test_dataset)} 样本")
        else:
            self.valid_dataset = None
            self.test_dataset = None
            logger.warning("测试数据集为空，无法创建验证集和测试集")

    def create_training_arguments(self, epochs=3, learning_rate=2e-5, **kwargs):
        """创建训练参数"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
            learning_rate=learning_rate,
            warmup_steps=kwargs.get('warmup_steps', 100),
            logging_steps=kwargs.get('logging_steps', 50),
            eval_strategy="epoch" if self.enable_evaluation and self.valid_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if self.enable_evaluation and self.valid_dataset else False,
            metric_for_best_model="eval_loss" if self.enable_evaluation and self.valid_dataset else None,
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=["labels"],
            max_grad_norm=kwargs.get('max_grad_norm', 1.0),
            optim="adamw_torch",
            weight_decay=kwargs.get('weight_decay', 0.01),
            lr_scheduler_type=kwargs.get('lr_scheduler_type', "cosine"),
            report_to="none",
            push_to_hub=False,
        )
        return training_args

    def train(self, training_args):
        """训练模型"""
        if self.train_dataset is None or self.data_collator is None:
            raise ValueError("请先调用 prepare_datasets() 方法准备训练数据")
        
        logger.info("正在初始化 Trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset if self.enable_evaluation else None,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 检查模型参数状态
        trainable_params = 0
        all_params = 0
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"总参数: {all_params:,}")
        logger.info(f"可训练参数比例: {trainable_params/all_params*100:.2f}%")
        
        logger.info("开始训练...")
        train_result = self.trainer.train()
        logger.info("训练完成")
        
        return train_result

    def evaluate_on_dataset(self, model, dataset, dataset_name, max_samples=None):
        """在指定数据集上评估模型"""
        logger.info(f"在{dataset_name}上评估...")
        
        if dataset is None:
            logger.warning(f"{dataset_name}为空，跳过评估")
            return None
        
        model.eval()
        total_loss = 0
        predictions = []
        ground_truths = []
        
        # 限制样本数量
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"限制{dataset_name}样本数量为: {max_samples}")
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"评估 {dataset_name}"):
                try:
                    # 获取单个样本
                    sample = dataset[i]
                    
                    # 转换为张量
                    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0)
                    attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0)
                    labels = torch.tensor(sample['labels']).unsqueeze(0)
                    
                    # 移动到正确的设备
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    # 前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss):
                        logger.warning(f"样本 {i} 的损失为NaN，跳过")
                        continue
                    
                    total_loss += loss.item()
                    
                    # 生成预测
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3
                    )
                    
                    # 解码预测结果
                    for j, generated_id in enumerate(generated_ids):
                        # 提取生成的部分
                        input_length = input_ids[j].shape[0]
                        generated_text = self.tokenizer.decode(
                            generated_id[input_length:], 
                            skip_special_tokens=True
                        )
                        
                        # 提取真实输出
                        full_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)
                        if "Output:" in full_text:
                            true_output = full_text.split("Output:")[-1].strip()
                        else:
                            true_output = ""
                        
                        predictions.append(generated_text)
                        ground_truths.append(true_output)
                        
                except Exception as e:
                    logger.error(f"处理样本 {i} 时出错: {e}")
                    continue
        
        # 检查是否有有效结果
        if len(predictions) == 0:
            logger.error("没有生成任何有效预测")
            return None
        
        # 计算基础指标
        avg_loss = total_loss / len(predictions) if total_loss > 0 else float('nan')
        
        # 简单的文本相似度指标
        exact_match = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
        exact_match_rate = exact_match / len(predictions) if predictions else 0
        
        # 计算部分匹配
        partial_match = 0
        for pred, truth in zip(predictions, ground_truths):
            pred_words = set(pred.strip().split())
            truth_words = set(truth.strip().split())
            if pred_words and truth_words:
                overlap = len(pred_words.intersection(truth_words))
                if overlap > 0:
                    partial_match += overlap / len(truth_words)
        
        partial_match_rate = partial_match / len(predictions) if predictions else 0
        
        # 计算BLEU和ROUGE指标
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        if BLEU_ROUGE_AVAILABLE:
            # 初始化ROUGE计算器
            rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            smoothing = SmoothingFunction().method1
            
            for pred, truth in zip(predictions, ground_truths):
                # 计算BLEU
                pred_tokens = pred.strip().split()
                truth_tokens = truth.strip().split()
                
                if pred_tokens and truth_tokens:
                    bleu_score = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(bleu_score)
                
                # 计算ROUGE
                if pred.strip() and truth.strip():
                    rouge_result = rouge_calculator.score(truth.strip(), pred.strip())
                    rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
        
        # 计算平均BLEU和ROUGE分数
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0
        avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0
        avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
        
        results = {
            'dataset_name': dataset_name,
            'avg_loss': avg_loss,
            'exact_match_rate': exact_match_rate,
            'partial_match_rate': partial_match_rate,
            'bleu_score': avg_bleu,
            'rouge1_score': avg_rouge1,
            'rouge2_score': avg_rouge2,
            'rougeL_score': avg_rougeL,
            'total_samples': len(predictions),
            'predictions': predictions[:10],
            'ground_truths': ground_truths[:10]
        }
        
        logger.info(f"{dataset_name} 评估结果:")
        logger.info(f"  平均损失: {avg_loss:.4f}")
        logger.info(f"  精确匹配率: {exact_match_rate:.4f}")
        logger.info(f"  部分匹配率: {partial_match_rate:.4f}")
        if BLEU_ROUGE_AVAILABLE:
            logger.info(f"  BLEU分数: {avg_bleu:.4f}")
            logger.info(f"  ROUGE-1分数: {avg_rouge1:.4f}")
            logger.info(f"  ROUGE-2分数: {avg_rouge2:.4f}")
            logger.info(f"  ROUGE-L分数: {avg_rougeL:.4f}")
        else:
            logger.info("  BLEU和ROUGE指标未计算（库未安装）")
        logger.info(f"  有效样本数: {len(predictions)}")
        
        return results

    def save_model(self, save_path=None):
        """保存模型"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "final_adapters")
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"正在保存微调后的 PEFT 适配器到: {save_path}")
        self.trainer.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("保存完成")

    def run_training_with_evaluation(self, epochs=3, learning_rate=2e-5, **kwargs):
        """运行带评估的训练流程"""
        logger.info("开始带评估的训练流程")
        
        try:
            # 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 配置PEFT
            self.configure_peft()
            
            # 准备数据集
            if not self.prepare_datasets():
                logger.error("数据集准备失败")
                return None
            
            # 创建训练参数
            training_args = self.create_training_arguments(epochs, learning_rate, **kwargs)
            
            # 训练模型
            train_result = self.train(training_args)
            
            # 保存模型
            self.save_model()
            
            # 评估模型
            validation_results = None
            test_results = None
            
            if self.valid_dataset:
                logger.info("在验证集上评估...")
                validation_results = self.evaluate_on_dataset(
                    self.model, self.valid_dataset, "验证集", max_samples=100
                )
            
            if self.test_dataset:
                logger.info("在测试集上评估...")
                test_results = self.evaluate_on_dataset(
                    self.model, self.test_dataset, "测试集", max_samples=100
                )
            
            # 保存结果
            all_results = {
                'train_result': train_result,
                'validation_results': validation_results,
                'test_results': test_results,
                'model_path': self.output_dir
            }
            
            results_file = os.path.join(self.output_dir, "evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # 打印总体统计
            self.print_overall_statistics(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"训练和评估过程中出错: {str(e)}")
            raise

    def run_training_only(self, epochs=3, learning_rate=2e-5, **kwargs):
        """运行仅训练流程（不评估）"""
        logger.info("开始仅训练流程")
        
        try:
            # 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 配置PEFT
            self.configure_peft()
            
            # 准备数据集
            if not self.prepare_datasets():
                logger.error("数据集准备失败")
                return None
            
            # 创建训练参数
            training_args = self.create_training_arguments(epochs, learning_rate, **kwargs)
            
            # 训练模型
            train_result = self.train(training_args)
            
            # 保存模型
            self.save_model()
            
            logger.info("仅训练流程完成")
            return train_result
            
        except Exception as e:
            logger.error(f"训练过程中出错: {str(e)}")
            raise

    def print_overall_statistics(self, results):
        """打印总体统计信息"""
        logger.info("\n" + "="*60)
        logger.info("总体训练和评估统计")
        logger.info("="*60)
        
        # 验证集结果
        if results.get('validation_results'):
            val_results = results['validation_results']
            logger.info(f"✅ 验证集结果:")
            logger.info(f"   平均损失: {val_results['avg_loss']:.4f}")
            logger.info(f"   精确匹配率: {val_results['exact_match_rate']:.4f}")
            logger.info(f"   部分匹配率: {val_results['partial_match_rate']:.4f}")
            if BLEU_ROUGE_AVAILABLE:
                logger.info(f"   BLEU分数: {val_results.get('bleu_score', 0):.4f}")
                logger.info(f"   ROUGE-1分数: {val_results.get('rouge1_score', 0):.4f}")
                logger.info(f"   ROUGE-2分数: {val_results.get('rouge2_score', 0):.4f}")
                logger.info(f"   ROUGE-L分数: {val_results.get('rougeL_score', 0):.4f}")
            logger.info(f"   样本数: {val_results['total_samples']}")
        
        # 测试集结果
        if results.get('test_results'):
            test_results = results['test_results']
            logger.info(f"✅ 测试集结果:")
            logger.info(f"   平均损失: {test_results['avg_loss']:.4f}")
            logger.info(f"   精确匹配率: {test_results['exact_match_rate']:.4f}")
            logger.info(f"   部分匹配率: {test_results['partial_match_rate']:.4f}")
            if BLEU_ROUGE_AVAILABLE:
                logger.info(f"   BLEU分数: {test_results.get('bleu_score', 0):.4f}")
                logger.info(f"   ROUGE-1分数: {test_results.get('rouge1_score', 0):.4f}")
                logger.info(f"   ROUGE-2分数: {test_results.get('rouge2_score', 0):.4f}")
                logger.info(f"   ROUGE-L分数: {test_results.get('rougeL_score', 0):.4f}")
            logger.info(f"   样本数: {test_results['total_samples']}")
        
        # 训练结果
        if results.get('train_result'):
            train_results = results['train_result']
            logger.info(f"✅ 训练结果:")
            logger.info(f"   训练损失: {train_results.training_loss:.4f}")
            logger.info(f"   全局步数: {train_results.global_step}")

def main():
    """主函数 - 演示两种模式的使用"""
    
    # 模式1: 仅训练（不评估）
    # logger.info("=== 训练 ===")
    # trainer_only = FineTuner(
    #     model_name="Qwen/Qwen3-0.6B",
    #     train_dataset_name="fm-universe/FM-alpaca",
    #     output_dir="fine_tuned_models_only",
    #     enable_evaluation=False
    # )
    
    # 运行仅训练流程
    # trainer_only.run_training_only(epochs=3, learning_rate=2e-5)
    
    # 模式2: 训练+评估
    logger.info("=== 训练+评估 ===")
    trainer_with_eval = FineTuner(
        model_name="Qwen/Qwen3-0.6B",
        train_dataset_name="fm-universe/FM-alpaca",
        test_dataset_name="fm-universe/FM-bench",
        output_dir="fine_tuned_models_with_eval",
        enable_evaluation=True
    )
    
    # 运行带评估的训练流程

    results = trainer_with_eval.run_training_with_evaluation(
        epochs=3, 
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4
    )

    trainer_with_eval.print_overall_statistics(results)
    
    logger.info("训练流程完成!")

if __name__ == "__main__":
    main() 