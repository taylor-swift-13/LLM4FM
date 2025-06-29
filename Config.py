

class Config:
    def __init__(self):
        # API 模型的配置
        self.use_api_model = True # 控制是使用 API 模型还是本地 Transformers 模型
        self.api_model:str = "gpt-4o" # API 模型的名称，例如 "gpt-4o"
        self.api_key:str = "sk-hfyQZDWdgyc4oQnDw4nvOh6KT1iDQ5EbNy9UjQwnMzBntefe" # 你的 OpenAI 或兼容 API 的密钥
        self.base_url:str = "https://yunwu.ai/v1" # API 的 base_url
        self.api_temperature = 0.7 # API 调用的温度参数
        self.api_top_p=0.9

        # 本地 Transformers 模型的配置
        self.local_model_name:str = "Qwen/Qwen3-0.6B" # 本地模型的名称
        self.fine_tuned_adapter:str = "./qwen3_alpaca_acsl_finetuned_results/final_adapters"
        self.local_temperature = 0.7
        self.local_top_p=0.9



        self.fine_tune = True
        self.think_mode_enabled = False # 控制是否保留 <think> 标签内容

        # RAG 模型的配置
        self.embed_model:str = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.top_k = 5



    

   