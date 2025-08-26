"""
模型导出系统测试

测试多格式量化功能、中文处理能力验证和模型导出功能。
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.model_exporter import (
    ModelQuantizer,
    ChineseCapabilityValidator,
    ModelExporter,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationBackend,
    QuantizationResult,
    ModelMetadata,
    DeploymentPackage
)


class MockModel(nn.Module):
    """模拟模型用于测试"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 50000)  # 模拟词汇表大小
    
    def forward(self, input_ids, **kwargs):
        x = self.linear1(input_ids.float())
        x = self.linear2(x)
        logits = self.output(x)
        return Mock(logits=logits)
    
    def generate(self, input_ids, **kwargs):
        # 模拟生成过程
        batch_size, seq_len = input_ids.shape
        max_length = kwargs.get('max_length', seq_len + 50)
        new_tokens = max_length - seq_len
        
        # 生成随机token
        generated = torch.randint(0, 50000, (batch_size, new_tokens))
        return torch.cat([input_ids, generated], dim=1)


class MockTokenizer:
    """模拟分词器用于测试"""
    
    def __init__(self):
        self.pad_token_id = 0
        self.vocab_size = 50000
    
    def __call__(self, text, **kwargs):
        # 模拟编码过程
        tokens = [1, 2, 3, 4, 5]  # 简单的token序列
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
    
    def encode(self, text):
        return [1, 2, 3, 4, 5]
    
    def decode(self, tokens, **kwargs):
        # 模拟解码过程
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 根据输入返回相应的中文文本
        if any(token > 1000 for token in tokens):
            return "这是一个包含密码学术语AES的中文回答。"
        else:
            return "简单的中文回答。"
    
    def __len__(self):
        return self.vocab_size


class TestQuantizationConfig:
    """测试量化配置"""
    
    def test_int8_config_creation(self):
        """测试INT8配置创建"""
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        assert config.format == QuantizationFormat.INT8
        assert config.backend == QuantizationBackend.BITSANDBYTES
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
    
    def test_int4_config_creation(self):
        """测试INT4配置创建"""
        config = QuantizationConfig(
            format=QuantizationFormat.INT4,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        assert config.format == QuantizationFormat.INT4
        assert config.backend == QuantizationBackend.BITSANDBYTES
        assert config.load_in_8bit is False
        assert config.load_in_4bit is True
    
    def test_gptq_config_creation(self):
        """测试GPTQ配置创建"""
        config = QuantizationConfig(
            format=QuantizationFormat.GPTQ,
            backend=QuantizationBackend.GPTQ,
            bits=4,
            group_size=128
        )
        
        assert config.format == QuantizationFormat.GPTQ
        assert config.backend == QuantizationBackend.GPTQ
        assert config.bits == 4
        assert config.group_size == 128


class TestModelQuantizer:
    """测试模型量化器"""
    
    @pytest.fixture
    def quantizer(self):
        """创建量化器实例"""
        return ModelQuantizer()
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        return MockModel()
    
    @pytest.fixture
    def mock_tokenizer(self):
        """创建模拟分词器"""
        return MockTokenizer()
    
    def test_calculate_model_size(self, quantizer, mock_model):
        """测试模型大小计算"""
        size_mb = quantizer._calculate_model_size(mock_model)
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_generate_calibration_data(self, quantizer, mock_tokenizer):
        """测试校准数据生成"""
        calibration_data = quantizer._generate_calibration_data(mock_tokenizer)
        
        assert isinstance(calibration_data, list)
        assert len(calibration_data) > 0
        assert all(isinstance(text, str) for text in calibration_data)
        assert any("AES" in text for text in calibration_data)
        # 检查是否包含中文字符（任何中文字符）
        assert any(any('\u4e00' <= char <= '\u9fff' for char in text) for text in calibration_data)
    
    @patch('src.model_exporter.QUANTIZATION_AVAILABLE', True)
    def test_pytorch_quantization(self, quantizer, mock_model, mock_tokenizer):
        """测试PyTorch原生量化"""
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        quantized_model, result = quantizer.quantize_model(
            mock_model, mock_tokenizer, config
        )
        
        assert result.success is True
        assert result.original_size_mb > 0
        assert result.quantized_size_mb > 0
        assert result.compression_ratio > 0
    
    def test_quantization_failure_handling(self, quantizer, mock_tokenizer):
        """测试量化失败处理"""
        # 使用无效模型触发异常
        invalid_model = None
        
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        quantized_model, result = quantizer.quantize_model(
            invalid_model, mock_tokenizer, config
        )
        
        assert result.success is False
        assert result.error_message is not None
        assert quantized_model is None
    
    def test_accuracy_preservation_evaluation(self, quantizer, mock_model, mock_tokenizer):
        """测试准确性保持评估"""
        accuracy = quantizer._evaluate_accuracy_preservation(
            mock_model, mock_model, mock_tokenizer
        )
        
        assert 0.0 <= accuracy <= 1.0
        # 相同模型应该有高相似度（或者默认值0.8）
        assert accuracy >= 0.8
    
    def test_inference_speedup_measurement(self, quantizer, mock_model, mock_tokenizer):
        """测试推理加速测量"""
        speedup = quantizer._measure_inference_speedup(
            mock_model, mock_model, mock_tokenizer
        )
        
        assert speedup > 0
        # 相同模型的加速比应该接近1
        assert 0.5 <= speedup <= 2.0


class TestChineseCapabilityValidator:
    """测试中文处理能力验证器"""
    
    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        return ChineseCapabilityValidator()
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        return MockModel()
    
    @pytest.fixture
    def mock_tokenizer(self):
        """创建模拟分词器"""
        return MockTokenizer()
    
    def test_get_default_test_cases(self, validator):
        """测试默认测试用例获取"""
        test_cases = validator._get_default_chinese_test_cases()
        
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        
        for case in test_cases:
            assert "input" in case
            assert "expected_keywords" in case
            assert "category" in case
            assert isinstance(case["expected_keywords"], list)
    
    def test_chinese_encoding_accuracy(self, validator, mock_model, mock_tokenizer):
        """测试中文编码准确性"""
        accuracy = validator._test_chinese_encoding(mock_model, mock_tokenizer)
        
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
    
    def test_crypto_terms_accuracy(self, validator, mock_model, mock_tokenizer):
        """测试密码学术语准确性"""
        accuracy = validator._test_crypto_terms(mock_model, mock_tokenizer)
        
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
    
    def test_thinking_structure_preservation(self, validator, mock_model, mock_tokenizer):
        """测试思考结构保持"""
        score = validator._test_thinking_structure(mock_model, mock_tokenizer)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_semantic_coherence(self, validator, mock_model, mock_tokenizer):
        """测试语义连贯性"""
        score = validator._test_semantic_coherence(mock_model, mock_tokenizer)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_single_case_evaluation(self, validator, mock_model, mock_tokenizer):
        """测试单个用例评估"""
        test_case = {
            "input": "什么是AES加密算法？",
            "expected_keywords": ["对称加密", "高级加密标准"],
            "category": "基础概念"
        }
        
        result = validator._evaluate_single_case(mock_model, mock_tokenizer, test_case)
        
        assert "input" in result
        assert "response" in result
        assert "score" in result
        assert "category" in result
        assert "success" in result
        assert 0.0 <= result["score"] <= 1.0
    
    def test_response_scoring(self, validator):
        """测试响应评分"""
        response = "AES是一种对称加密算法，广泛应用于数据保护。"
        expected_keywords = ["对称加密", "数据保护"]
        
        score = validator._score_response(response, expected_keywords)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # 应该有较高得分
    
    def test_char_accuracy_calculation(self, validator):
        """测试字符准确性计算"""
        original = "这是一个测试文本"
        decoded = "这是一个测试文本"
        
        accuracy = validator._calculate_char_accuracy(original, decoded)
        assert accuracy == 1.0
        
        # 测试部分匹配
        decoded_partial = "这是一个测试"
        accuracy_partial = validator._calculate_char_accuracy(original, decoded_partial)
        assert 0.0 < accuracy_partial < 1.0
    
    def test_validate_chinese_capability(self, validator, mock_model, mock_tokenizer):
        """测试中文能力验证"""
        results = validator.validate_chinese_capability(mock_model, mock_tokenizer)
        
        assert "overall_score" in results
        assert "test_results" in results
        assert "chinese_encoding_accuracy" in results
        assert "crypto_term_accuracy" in results
        assert "thinking_structure_preservation" in results
        assert "semantic_coherence" in results
        
        assert 0.0 <= results["overall_score"] <= 1.0
        assert isinstance(results["test_results"], list)


class TestModelExporter:
    """测试模型导出器"""
    
    @pytest.fixture
    def exporter(self):
        """创建导出器实例"""
        return ModelExporter()
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        model = MockModel()
        # 添加save_pretrained方法
        model.save_pretrained = Mock()
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """创建模拟分词器"""
        tokenizer = MockTokenizer()
        tokenizer.save_pretrained = Mock()
        return tokenizer
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_generate_metadata(self, exporter):
        """测试元数据生成"""
        quant_result = QuantizationResult(
            success=True,
            original_size_mb=1000.0,
            quantized_size_mb=500.0,
            compression_ratio=2.0,
            accuracy_preserved=0.95,
            inference_speedup=1.5,
            memory_reduction=0.5,
            quantization_time=60.0
        )
        
        chinese_validation = {
            "overall_score": 0.9,
            "crypto_term_accuracy": 0.85
        }
        
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        metadata = exporter._generate_metadata(
            "test-model", quant_result, chinese_validation, config
        )
        
        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_name == "test-model"
        assert metadata.quantization_format == "int8"
        assert metadata.compression_ratio == 2.0
        assert len(metadata.supported_languages) > 0
        assert len(metadata.specialized_domains) > 0
    
    def test_generate_readme(self, exporter, temp_dir):
        """测试README生成"""
        metadata = ModelMetadata(
            model_name="test-model",
            model_version="1.0.0",
            quantization_format="int8",
            original_model_size=1000.0,
            quantized_model_size=500.0,
            compression_ratio=2.0,
            supported_languages=["中文"],
            specialized_domains=["密码学"],
            creation_time="2024-01-01T00:00:00",
            framework_version="2.0.0",
            python_version="3.8.0",
            hardware_requirements={"min_gpu_memory_gb": 4},
            usage_instructions="测试说明",
            performance_metrics={
                "compression_ratio": 2.0,
                "inference_speedup": 1.5,
                "memory_reduction": 0.5,
                "chinese_capability_score": 0.9,
                "crypto_term_accuracy": 0.85
            }
        )
        
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        chinese_validation = {"overall_score": 0.9}
        
        readme_path = Path(temp_dir) / "README.md"
        exporter._generate_readme(readme_path, metadata, config, chinese_validation)
        
        assert readme_path.exists()
        
        content = readme_path.read_text(encoding='utf-8')
        assert "test-model" in content
        assert "量化格式" in content
        assert "中文能力得分" in content
        assert "安装和使用" in content
    
    def test_generate_requirements(self, exporter, temp_dir):
        """测试requirements.txt生成"""
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        requirements_path = Path(temp_dir) / "requirements.txt"
        exporter._generate_requirements(requirements_path, config)
        
        assert requirements_path.exists()
        
        content = requirements_path.read_text(encoding='utf-8')
        assert "torch" in content
        assert "transformers" in content
        assert "bitsandbytes" in content
    
    def test_calculate_directory_size(self, exporter, temp_dir):
        """测试目录大小计算"""
        # 创建一些测试文件
        test_file1 = Path(temp_dir) / "test1.txt"
        test_file2 = Path(temp_dir) / "test2.txt"
        
        test_file1.write_text("Hello World" * 100)
        test_file2.write_text("Test Content" * 50)
        
        size_mb = exporter._calculate_directory_size(Path(temp_dir))
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_generate_checksum(self, exporter, temp_dir):
        """测试校验和生成"""
        # 创建测试文件
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content for checksum")
        
        checksum = exporter._generate_checksum(Path(temp_dir))
        
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5哈希长度
    
    @patch.object(ModelQuantizer, 'quantize_model')
    @patch.object(ChineseCapabilityValidator, 'validate_chinese_capability')
    def test_export_quantized_model(
        self,
        mock_validate,
        mock_quantize,
        exporter,
        mock_model,
        mock_tokenizer,
        temp_dir
    ):
        """测试量化模型导出"""
        # 设置mock返回值
        mock_quantize.return_value = (
            mock_model,
            QuantizationResult(
                success=True,
                original_size_mb=1000.0,
                quantized_size_mb=500.0,
                compression_ratio=2.0,
                accuracy_preserved=0.95,
                inference_speedup=1.5,
                memory_reduction=0.5,
                quantization_time=60.0
            )
        )
        
        mock_validate.return_value = {
            "overall_score": 0.9,
            "crypto_term_accuracy": 0.85,
            "chinese_encoding_accuracy": 0.9,
            "thinking_structure_preservation": 0.8,
            "semantic_coherence": 0.85
        }
        
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        # 执行导出
        deployment_package = exporter.export_quantized_model(
            mock_model,
            mock_tokenizer,
            temp_dir,
            config,
            "test-model"
        )
        
        # 验证结果
        assert isinstance(deployment_package, DeploymentPackage)
        assert deployment_package.package_path == temp_dir
        assert len(deployment_package.model_files) > 0
        assert len(deployment_package.config_files) > 0
        assert deployment_package.package_size_mb > 0
        assert deployment_package.checksum is not None
        
        # 验证文件存在
        output_path = Path(temp_dir)
        assert (output_path / "model").exists()
        assert (output_path / "metadata.json").exists()
        assert (output_path / "README.md").exists()
        assert (output_path / "requirements.txt").exists()
        assert (output_path / "quantization_config.json").exists()
        assert (output_path / "validation_results.json").exists()
        
        # 验证配置文件内容
        with open(output_path / "quantization_config.json", 'r', encoding='utf-8') as f:
            quant_config = json.load(f)
            assert quant_config["format"] == "int8"
            assert quant_config["backend"] == "bitsandbytes"
        
        # 验证验证结果文件
        with open(output_path / "validation_results.json", 'r', encoding='utf-8') as f:
            validation_results = json.load(f)
            assert validation_results["overall_score"] == 0.9
    
    @patch.object(ModelQuantizer, 'quantize_model')
    def test_export_with_quantization_failure(
        self,
        mock_quantize,
        exporter,
        mock_model,
        mock_tokenizer,
        temp_dir
    ):
        """测试量化失败时的导出处理"""
        # 设置量化失败
        mock_quantize.return_value = (
            mock_model,
            QuantizationResult(
                success=False,
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=0,
                accuracy_preserved=0,
                inference_speedup=0,
                memory_reduction=0,
                quantization_time=0,
                error_message="量化失败测试"
            )
        )
        
        config = QuantizationConfig(
            format=QuantizationFormat.INT8,
            backend=QuantizationBackend.BITSANDBYTES
        )
        
        # 应该抛出异常
        with pytest.raises(RuntimeError, match="模型量化失败"):
            exporter.export_quantized_model(
                mock_model,
                mock_tokenizer,
                temp_dir,
                config
            )


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_quantization_workflow(self):
        """测试端到端量化工作流"""
        # 创建简单的测试模型
        model = MockModel()
        tokenizer = MockTokenizer()
        
        # 创建量化配置
        config = QuantizationConfig(
            format=QuantizationFormat.DYNAMIC,
            backend=QuantizationBackend.PYTORCH
        )
        
        # 执行量化
        quantizer = ModelQuantizer()
        quantized_model, result = quantizer.quantize_model(model, tokenizer, config)
        
        # 验证结果
        assert result.success is True
        assert quantized_model is not None
        assert result.compression_ratio > 0
    
    def test_chinese_capability_validation_workflow(self):
        """测试中文能力验证工作流"""
        model = MockModel()
        tokenizer = MockTokenizer()
        
        validator = ChineseCapabilityValidator()
        results = validator.validate_chinese_capability(model, tokenizer)
        
        assert results["overall_score"] >= 0
        assert len(results["test_results"]) > 0
        assert all(key in results for key in [
            "chinese_encoding_accuracy",
            "crypto_term_accuracy",
            "thinking_structure_preservation",
            "semantic_coherence"
        ])


if __name__ == "__main__":
    pytest.main([__file__])