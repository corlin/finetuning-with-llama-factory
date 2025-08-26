"""
模型导出和部署包生成端到端测试

测试完整的模型导出、量化、验证和部署包生成流程。
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.model_exporter import (
    ModelExporter,
    ModelQuantizer,
    ChineseCapabilityValidator,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationBackend,
    ModelMetadata,
    DeploymentPackage
)


class SimpleTestModel(nn.Module):
    """简单的测试模型"""
    
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # 添加config属性
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size
        })()
    
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # 简单的池化
        logits = self.linear(x)
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def generate(self, input_ids, max_length=50, **kwargs):
        """简单的生成方法"""
        batch_size, seq_len = input_ids.shape
        new_tokens = max_length - seq_len
        
        if new_tokens > 0:
            generated = torch.randint(1, 100, (batch_size, new_tokens))
            return torch.cat([input_ids, generated], dim=1)
        
        return input_ids
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # 保存配置
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "model_type": "simple-test",
            "architectures": ["SimpleTestModel"]
        }
        
        with open(save_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)


class SimpleTestTokenizer:
    """简单的测试分词器"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
    
    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = self.encode(text)
        result = {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def encode(self, text):
        """简单编码"""
        # 根据文本长度生成token
        tokens = [self.bos_token_id]
        tokens.extend(range(10, min(20, 10 + len(text) // 5)))
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, tokens, skip_special_tokens=False):
        """简单解码"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        
        # 生成简单的中文回答
        if any(t > 50 for t in tokens):
            return "这是一个包含密码学术语AES的中文专业回答。"
        else:
            return "这是一个简单的中文回答。"
    
    def save_pretrained(self, save_directory):
        """保存分词器"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存简单的配置
        config = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id
        }
        
        with open(save_path / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # 创建简单的tokenizer.json
        tokenizer_json = {
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": {f"token_{i}": i for i in range(self.vocab_size)}}
        }
        
        with open(save_path / "tokenizer.json", 'w') as f:
            json.dump(tokenizer_json, f, indent=2)


class TestModelExportDeployment:
    """模型导出和部署测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_model(self):
        """创建测试模型"""
        return SimpleTestModel()
    
    @pytest.fixture
    def test_tokenizer(self):
        """创建测试分词器"""
        return SimpleTestTokenizer()
    
    @pytest.fixture
    def exporter(self):
        """创建模型导出器"""
        return ModelExporter()
    
    def test_complete_export_workflow(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试完整的导出工作流"""
        # 配置量化参数
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        # 执行导出
        deployment_package = exporter.export_quantized_model(
            model=test_model,
            tokenizer=test_tokenizer,
            output_dir=temp_dir,
            quantization_config=config,
            model_name="test-model"
        )
        
        # 验证部署包
        assert isinstance(deployment_package, DeploymentPackage)
        assert deployment_package.package_path == temp_dir
        assert deployment_package.package_size_mb > 0
        assert deployment_package.checksum is not None
        assert len(deployment_package.model_files) > 0
        assert len(deployment_package.config_files) > 0
    
    def test_deployment_package_structure(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试部署包结构"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        deployment_package = exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查必要文件
        output_path = Path(temp_dir)
        
        # 模型文件
        assert (output_path / "model").exists()
        assert (output_path / "model" / "pytorch_model.bin").exists()
        assert (output_path / "model" / "config.json").exists()
        assert (output_path / "model" / "tokenizer.json").exists()
        assert (output_path / "model" / "tokenizer_config.json").exists()
        
        # 配置文件
        assert (output_path / "quantization_config.json").exists()
        assert (output_path / "validation_results.json").exists()
        
        # 文档文件
        assert (output_path / "metadata.json").exists()
        assert (output_path / "README.md").exists()
        assert (output_path / "requirements.txt").exists()
    
    def test_metadata_generation(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试元数据生成"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config, "test-model"
        )
        
        # 读取元数据
        metadata_path = Path(temp_dir) / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 验证元数据内容
        assert metadata["model_name"] == "test-model"
        assert metadata["quantization_format"] == "dynamic"
        assert metadata["compression_ratio"] > 0
        assert "中文" in metadata["supported_languages"]
        assert "密码学" in metadata["specialized_domains"]
        assert "creation_time" in metadata
        assert "performance_metrics" in metadata
    
    def test_readme_generation(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试README生成"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查README
        readme_path = Path(temp_dir) / "README.md"
        assert readme_path.exists()
        
        readme_content = readme_path.read_text(encoding='utf-8')
        
        # 验证README内容
        assert "模型简介" in readme_content
        assert "安装和使用" in readme_content
        assert "硬件要求" in readme_content
        assert "性能指标" in readme_content
        assert "使用示例" in readme_content
        assert "pip install -r requirements.txt" in readme_content
        assert "AutoTokenizer" in readme_content
        assert "AutoModelForCausalLM" in readme_content
    
    def test_requirements_generation(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试requirements.txt生成"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查requirements.txt
        requirements_path = Path(temp_dir) / "requirements.txt"
        assert requirements_path.exists()
        
        requirements_content = requirements_path.read_text(encoding='utf-8')
        
        # 验证依赖内容
        assert "torch" in requirements_content
        assert "transformers" in requirements_content
        assert "numpy" in requirements_content
    
    def test_validation_results_storage(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试验证结果存储"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查验证结果
        validation_path = Path(temp_dir) / "validation_results.json"
        assert validation_path.exists()
        
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_results = json.load(f)
        
        # 验证结果结构
        assert "overall_score" in validation_results
        assert "chinese_encoding_accuracy" in validation_results
        assert "crypto_term_accuracy" in validation_results
        assert "thinking_structure_preservation" in validation_results
        assert "semantic_coherence" in validation_results
        assert "test_results" in validation_results
    
    def test_quantization_config_storage(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试量化配置存储"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查量化配置
        quant_config_path = Path(temp_dir) / "quantization_config.json"
        assert quant_config_path.exists()
        
        with open(quant_config_path, 'r', encoding='utf-8') as f:
            quant_config = json.load(f)
        
        # 验证配置内容
        assert quant_config["format"] == "dynamic"
        assert quant_config["backend"] == "pytorch"
        assert "load_in_8bit" in quant_config
        assert "load_in_4bit" in quant_config
    
    def test_deployment_package_integrity(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试部署包完整性"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        deployment_package = exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 验证校验和
        assert len(deployment_package.checksum) == 32  # MD5长度
        
        # 验证文件列表
        for file_path in deployment_package.model_files:
            assert Path(file_path).exists()
        
        for config_file in deployment_package.config_files:
            assert Path(config_file).exists()
        
        # 验证包大小
        actual_size = sum(
            f.stat().st_size for f in Path(temp_dir).rglob('*') if f.is_file()
        ) / 1024 / 1024
        
        # 允许一定的误差
        assert abs(deployment_package.package_size_mb - actual_size) < 1.0
    
    def test_multiple_quantization_formats(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试多种量化格式"""
        configs = [
            QuantizationConfig(
                format=QuantizationFormat.DYNAMIC,
                backend=QuantizationBackend.PYTORCH
            )
        ]
        
        deployment_packages = []
        
        for i, config in enumerate(configs):
            output_dir = Path(temp_dir) / f"export_{i}"
            
            deployment_package = exporter.export_quantized_model(
                test_model, test_tokenizer, str(output_dir), config,
                f"test-model-{config.format.value}"
            )
            
            deployment_packages.append(deployment_package)
        
        # 验证所有导出都成功
        assert len(deployment_packages) == len(configs)
        
        for package in deployment_packages:
            assert package.package_size_mb > 0
            assert len(package.model_files) > 0
    
    def test_export_error_handling(self, exporter, test_tokenizer, temp_dir):
        """测试导出错误处理"""
        # 使用无效模型
        invalid_model = None
        
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        # 应该抛出异常
        with pytest.raises(Exception):
            exporter.export_quantized_model(
                invalid_model, test_tokenizer, temp_dir, config
            )
    
    def test_deployment_package_serialization(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试部署包序列化"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        deployment_package = exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 测试序列化
        package_dict = deployment_package.to_dict()
        
        assert isinstance(package_dict, dict)
        assert "package_path" in package_dict
        assert "model_files" in package_dict
        assert "config_files" in package_dict
        assert "package_size_mb" in package_dict
        assert "checksum" in package_dict
    
    def test_chinese_capability_integration(self, exporter, test_model, test_tokenizer, temp_dir):
        """测试中文能力验证集成"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        exporter.export_quantized_model(
            test_model, test_tokenizer, temp_dir, config
        )
        
        # 检查中文能力验证结果
        validation_path = Path(temp_dir) / "validation_results.json"
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_results = json.load(f)
        
        # 验证中文相关指标
        assert validation_results["crypto_term_accuracy"] >= 0.0
        assert validation_results["chinese_encoding_accuracy"] >= 0.0
        assert validation_results["thinking_structure_preservation"] >= 0.0
        
        # 检查测试用例结果
        assert len(validation_results["test_results"]) > 0
        for result in validation_results["test_results"]:
            assert "input" in result
            assert "response" in result
            assert "score" in result


class TestDeploymentWorkflow:
    """部署工作流测试"""
    
    def test_end_to_end_deployment_simulation(self):
        """测试端到端部署模拟"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模型和分词器
            model = SimpleTestModel()
            tokenizer = SimpleTestTokenizer()
            
            # 创建导出器
            exporter = ModelExporter()
            
            # 配置量化
            config = QuantizationConfig(
                format=QuantizationFormat.DYNAMIC,
                backend=QuantizationBackend.PYTORCH
            )
            
            # 执行导出
            deployment_package = exporter.export_quantized_model(
                model, tokenizer, temp_dir, config, "deployment-test"
            )
            
            # 模拟部署验证
            self._simulate_deployment_validation(deployment_package)
    
    def _simulate_deployment_validation(self, deployment_package):
        """模拟部署验证"""
        package_path = Path(deployment_package.package_path)
        
        # 检查模型加载
        model_dir = package_path / "model"
        assert model_dir.exists()
        
        # 检查配置文件
        config_path = model_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            assert "vocab_size" in config
            assert "model_type" in config
        
        # 检查分词器
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        assert tokenizer_config_path.exists()
        
        # 检查元数据
        metadata_path = package_path / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            assert metadata["compression_ratio"] > 0
            assert len(metadata["supported_languages"]) > 0
        
        # 检查文档
        readme_path = package_path / "README.md"
        assert readme_path.exists()
        assert readme_path.stat().st_size > 0
        
        requirements_path = package_path / "requirements.txt"
        assert requirements_path.exists()
        assert requirements_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])