"""
LLaMA Factory适配器测试

测试LLaMA Factory配置文件生成器、数据格式转换和模型配置映射功能。
"""

import pytest
import json
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamafactory_adapter import (
    LlamaFactoryAdapter, LlamaFactoryDataConverter, LlamaFactoryConfigGenerator,
    LlamaFactoryModelConfig, LlamaFactoryLoRAConfig, LlamaFactoryTrainingConfig
)
from data_models import TrainingExample, ThinkingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig
from lora_config_optimizer import LoRAMemoryProfile
from parallel_config import ParallelConfig


class TestLlamaFactoryDataConverter:
    """测试数据格式转换器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.converter = LlamaFactoryDataConverter()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.test_examples = [
            TrainingExample(
                instruction="什么是AES加密算法？",
                input="",
                output="AES是高级加密标准，是一种对称分组密码算法。",
                thinking="<thinking>用户询问AES的基本概念，我需要简洁地解释其定义和特点。</thinking>",
                crypto_terms=["AES", "对称加密", "分组密码"],
                difficulty_level=DifficultyLevel.BEGINNER
            ),
            TrainingExample(
                instruction="比较AES和DES的安全性",
                input="在现代密码学应用中",
                output="AES比DES更安全，密钥长度更长，抗攻击能力更强。",
                crypto_terms=["AES", "DES", "安全性"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
        ]
        
        self.thinking_examples = [
            ThinkingExample(
                instruction="分析RSA算法的安全性",
                thinking_process="首先需要了解RSA的数学基础，然后分析其安全性依赖...",
                final_response="RSA算法的安全性基于大整数分解的困难性。",
                crypto_terms=["RSA", "非对称加密", "安全性"],
                difficulty_level=DifficultyLevel.ADVANCED
            )
        ]
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_convert_to_alpaca_format(self):
        """测试转换为Alpaca格式"""
        output_file = Path(self.temp_dir) / "test_alpaca.json"
        
        success = self.converter.convert_training_examples(
            self.test_examples, str(output_file), "alpaca"
        )
        
        assert success
        assert output_file.exists()
        
        # 验证转换结果
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert "instruction" in data[0]
        assert "input" in data[0]
        assert "output" in data[0]
        assert "system" in data[0]
        
        # 验证thinking数据处理
        assert "<thinking>" in data[0]["output"]
        assert "</thinking>" in data[0]["output"]
    
    def test_convert_to_sharegpt_format(self):
        """测试转换为ShareGPT格式"""
        output_file = Path(self.temp_dir) / "test_sharegpt.json"
        
        success = self.converter.convert_training_examples(
            self.test_examples, str(output_file), "sharegpt"
        )
        
        assert success
        assert output_file.exists()
        
        # 验证转换结果
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert "conversations" in data[0]
        
        conversations = data[0]["conversations"]
        assert len(conversations) == 3  # system, human, gpt
        assert conversations[0]["from"] == "system"
        assert conversations[1]["from"] == "human"
        assert conversations[2]["from"] == "gpt"
    
    def test_convert_thinking_examples(self):
        """测试转换思考样例"""
        output_file = Path(self.temp_dir) / "test_thinking.json"
        
        success = self.converter.convert_thinking_examples(
            self.thinking_examples, str(output_file), "alpaca"
        )
        
        assert success
        assert output_file.exists()
        
        # 验证转换结果
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert "<thinking>" in data[0]["output"]
    
    def test_validate_converted_data(self):
        """测试验证转换后的数据"""
        # 先转换数据
        output_file = Path(self.temp_dir) / "test_validate.json"
        self.converter.convert_training_examples(
            self.test_examples, str(output_file), "alpaca"
        )
        
        # 验证数据
        result = self.converter.validate_converted_data(str(output_file), "alpaca")
        
        assert result["valid"]
        assert len(result["errors"]) == 0
        assert result["statistics"]["total_samples"] == 2
        assert result["statistics"]["thinking_samples"] == 1
    
    def test_validate_invalid_data(self):
        """测试验证无效数据"""
        # 创建无效数据文件
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump([{"invalid": "data"}], f)
        
        result = self.converter.validate_converted_data(str(invalid_file), "alpaca")
        
        assert not result["valid"]
        assert len(result["errors"]) > 0


class TestLlamaFactoryConfigGenerator:
    """测试配置生成器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.generator = LlamaFactoryConfigGenerator()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.training_config = TrainingConfig(
            output_dir="./test_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-4
        )
        
        self.data_config = DataConfig(
            train_data_path="./data/train.json",
            eval_data_path="./data/eval.json"
        )
        
        self.lora_config = LoRAMemoryProfile(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        self.parallel_config = ParallelConfig(
            world_size=1,
            data_parallel_size=1
        )
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_training_config(self):
        """测试生成训练配置"""
        config_file = self.generator.generate_training_config(
            self.training_config,
            self.data_config,
            self.lora_config,
            self.parallel_config,
            "test_dataset",
            self.temp_dir
        )
        
        assert Path(config_file).exists()
        
        # 验证配置内容
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert config["model_name"] == "Qwen/Qwen3-4B-Thinking-2507"
        assert config["dataset"] == "test_dataset"
        assert config["lora_rank"] == 16
        assert config["lora_alpha"] == 32
        assert config["learning_rate"] == 2e-4
        assert config["num_train_epochs"] == 3
    
    def test_generate_dataset_info(self):
        """测试生成数据集信息"""
        info_file = self.generator.generate_dataset_info(
            "test_dataset",
            "train.json",
            "val.json",
            "alpaca",
            self.temp_dir
        )
        
        assert Path(info_file).exists()
        
        # 验证信息内容
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        assert "test_dataset" in info
        assert info["test_dataset"]["formatting"] == "alpaca"
        assert info["test_dataset"]["file_name"] == ["train.json", "val.json"]
    
    def test_generate_dataset_info_single_file(self):
        """测试生成单文件数据集信息"""
        info_file = self.generator.generate_dataset_info(
            "single_dataset",
            "train.json",
            None,
            "alpaca",
            self.temp_dir
        )
        
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        assert info["single_dataset"]["file_name"] == "train.json"


class TestLlamaFactoryAdapter:
    """测试LLaMA Factory适配器主类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.adapter = LlamaFactoryAdapter()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.test_examples = [
            TrainingExample(
                instruction="解释对称加密的原理",
                input="",
                output="对称加密使用相同的密钥进行加密和解密。",
                thinking="<thinking>需要解释对称加密的基本概念和工作原理。</thinking>",
                crypto_terms=["对称加密", "密钥"],
                difficulty_level=DifficultyLevel.BEGINNER
            ),
            TrainingExample(
                instruction="比较AES和ChaCha20算法",
                input="在移动设备上的应用",
                output="AES适用于硬件加速，ChaCha20适用于软件实现。",
                crypto_terms=["AES", "ChaCha20"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
        ]
        
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.lora_config = LoRAMemoryProfile(
            rank=8, alpha=16, target_modules=["q_proj", "v_proj"]
        )
        self.parallel_config = ParallelConfig()
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_training_data(self):
        """测试准备训练数据"""
        result = self.adapter.prepare_training_data(
            self.test_examples,
            self.temp_dir,
            "test_crypto",
            "alpaca",
            0.8
        )
        
        assert "train_file" in result
        assert "val_file" in result
        assert "dataset_info_file" in result
        
        # 验证文件存在
        assert Path(result["train_file"]).exists()
        assert Path(result["val_file"]).exists()
        assert Path(result["dataset_info_file"]).exists()
        
        # 验证数据分割
        with open(result["train_file"], 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(result["val_file"], 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        assert len(train_data) == 1  # 80% of 2 examples
        assert len(val_data) == 1   # 20% of 2 examples
    
    def test_create_training_config(self):
        """测试创建训练配置"""
        config_file = self.adapter.create_training_config(
            self.training_config,
            self.data_config,
            self.lora_config,
            self.parallel_config,
            "test_dataset",
            self.temp_dir
        )
        
        assert Path(config_file).exists()
        
        # 验证配置内容
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert config["dataset"] == "test_dataset"
        assert config["lora_rank"] == 8
        assert config["lora_alpha"] == 16
    
    def test_validate_integration(self):
        """测试验证集成"""
        # 准备数据和配置
        data_files = self.adapter.prepare_training_data(
            self.test_examples,
            self.temp_dir,
            "test_crypto"
        )
        
        config_file = self.adapter.create_training_config(
            self.training_config,
            self.data_config,
            self.lora_config,
            self.parallel_config,
            "test_crypto",
            self.temp_dir
        )
        
        # 验证集成
        result = self.adapter.validate_integration(config_file, data_files)
        
        assert result["valid"]
        assert len(result["errors"]) == 0
        assert "config_validation" in result
        assert "data_validation" in result
    
    def test_generate_training_script(self):
        """测试生成训练脚本"""
        config_file = "test_config.yaml"
        script_file = Path(self.temp_dir) / "train_script.py"
        
        generated_script = self.adapter.generate_training_script(
            config_file, str(script_file)
        )
        
        assert Path(generated_script).exists()
        
        # 验证脚本内容
        with open(generated_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "llamafactory.train.tuner import run_exp" in content
        assert config_file in content
        assert "def main():" in content


class TestLlamaFactoryConfigs:
    """测试LLaMA Factory配置类"""
    
    def test_model_config(self):
        """测试模型配置"""
        config = LlamaFactoryModelConfig(
            model_name="test_model",
            quantization_bit=4,
            flash_attn="fa2"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "test_model"
        assert config_dict["quantization_bit"] == 4
        assert config_dict["flash_attn"] == "fa2"
        assert "model_revision" in config_dict
    
    def test_lora_config(self):
        """测试LoRA配置"""
        config = LlamaFactoryLoRAConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_target="all"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["lora_rank"] == 16
        assert config_dict["lora_alpha"] == 32
        assert config_dict["lora_dropout"] == 0.1
        assert config_dict["lora_target"] == "all"
    
    def test_training_config(self):
        """测试训练配置"""
        config = LlamaFactoryTrainingConfig(
            dataset="test_dataset",
            learning_rate=1e-4,
            num_train_epochs=5,
            per_device_train_batch_size=2
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["dataset"] == "test_dataset"
        assert config_dict["learning_rate"] == 1e-4
        assert config_dict["num_train_epochs"] == 5
        assert config_dict["per_device_train_batch_size"] == 2


def test_integration_with_real_data():
    """集成测试：使用真实数据测试完整流程"""
    adapter = LlamaFactoryAdapter()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建更复杂的测试数据
        examples = []
        for i in range(10):
            example = TrainingExample(
                instruction=f"密码学问题 {i+1}",
                input=f"上下文信息 {i+1}",
                output=f"这是第{i+1}个回答，包含密码学知识。",
                thinking=f"<thinking>这是第{i+1}个思考过程，需要分析密码学概念。</thinking>",
                crypto_terms=["加密", "安全性", "算法"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
            examples.append(example)
        
        # 准备数据
        data_files = adapter.prepare_training_data(
            examples, temp_dir, "integration_test", "alpaca", 0.8
        )
        
        # 创建配置
        training_config = TrainingConfig(learning_rate=1e-4, num_train_epochs=2)
        data_config = DataConfig()
        lora_config = LoRAMemoryProfile(rank=8, alpha=16, target_modules=["q_proj"])
        parallel_config = ParallelConfig()
        
        config_file = adapter.create_training_config(
            training_config, data_config, lora_config, parallel_config,
            "integration_test", temp_dir
        )
        
        # 验证集成
        validation = adapter.validate_integration(config_file, data_files)
        
        assert validation["valid"]
        assert validation["data_validation"]["train_file"]["valid"]
        assert validation["data_validation"]["val_file"]["valid"]
        
        # 生成训练脚本
        script_file = adapter.generate_training_script(
            config_file, str(Path(temp_dir) / "train.py")
        )
        
        assert Path(script_file).exists()
        
        print(f"集成测试成功完成，文件保存在: {temp_dir}")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # 运行集成测试
    test_integration_with_real_data()
    print("所有测试通过！")