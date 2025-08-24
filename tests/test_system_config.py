"""
系统配置管理单元测试

测试系统配置的加载、验证、保存和管理功能。
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.system_config import (
    SystemConfig, ConfigManager, EnvironmentConfig, ModelConfig,
    TrainingConfig, DataConfig, ConfigFormat, LogLevel
)
from src.parallel_config import ParallelStrategy, CommunicationBackend, ZeroStage


class TestEnvironmentConfig:
    """环境配置测试"""
    
    def test_environment_config_creation(self):
        """测试环境配置创建"""
        config = EnvironmentConfig(
            python_version="3.12+",
            cuda_version="12.9",
            master_addr="192.168.1.100",
            master_port=29500
        )
        
        assert config.python_version == "3.12+"
        assert config.cuda_version == "12.9"
        assert config.master_addr == "192.168.1.100"
        assert config.master_port == 29500
    
    @patch.dict('os.environ', {
        'CUDA_VISIBLE_DEVICES': '0,1',
        'MASTER_ADDR': '192.168.1.200',
        'MASTER_PORT': '29501',
        'WORLD_SIZE': '2',
        'RANK': '1'
    })
    def test_environment_config_from_env(self):
        """测试从环境变量读取配置"""
        config = EnvironmentConfig()
        
        assert config.cuda_visible_devices == "0,1"
        assert config.master_addr == "192.168.1.200"
        assert config.master_port == 29501
        assert config.world_size == 2
        assert config.rank == 1
    
    def test_environment_config_serialization(self):
        """测试环境配置序列化"""
        config = EnvironmentConfig(
            python_version="3.12+",
            cuda_version="12.9"
        )
        
        data = config.to_dict()
        assert data["python_version"] == "3.12+"
        assert data["cuda_version"] == "12.9"


class TestModelConfig:
    """模型配置测试"""
    
    def test_model_config_creation(self):
        """测试模型配置创建"""
        config = ModelConfig(
            model_name="Qwen/Qwen3-4B-Thinking-2507",
            max_seq_length=2048,
            trust_remote_code=True
        )
        
        assert config.model_name == "Qwen/Qwen3-4B-Thinking-2507"
        assert config.max_seq_length == 2048
        assert config.trust_remote_code is True
    
    def test_model_config_validation(self):
        """测试模型配置验证"""
        # 测试无效的序列长度
        with pytest.raises(ValueError, match="max_seq_length必须大于0"):
            ModelConfig(max_seq_length=0)
        
        # 测试无效的padding_side
        with pytest.raises(ValueError, match="padding_side必须是'left'或'right'"):
            ModelConfig(padding_side="center")
    
    def test_model_config_serialization(self):
        """测试模型配置序列化"""
        config = ModelConfig(
            model_name="test_model",
            max_seq_length=1024
        )
        
        data = config.to_dict()
        assert data["model_name"] == "test_model"
        assert data["max_seq_length"] == 1024


class TestTrainingConfig:
    """训练配置测试"""
    
    def test_training_config_creation(self):
        """测试训练配置创建"""
        config = TrainingConfig(
            num_epochs=5,
            batch_size=8,
            per_device_batch_size=2,
            learning_rate=1e-4,
            use_lora=True,
            lora_rank=32
        )
        
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.use_lora is True
        assert config.lora_rank == 32
    
    def test_training_config_validation(self):
        """测试训练配置验证"""
        # 测试无效的epochs
        with pytest.raises(ValueError, match="num_epochs必须大于0"):
            TrainingConfig(num_epochs=0)
        
        # 测试无效的batch_size
        with pytest.raises(ValueError, match="batch_size必须大于0"):
            TrainingConfig(batch_size=0)
        
        # 测试无效的学习率
        with pytest.raises(ValueError, match="learning_rate必须在0-1之间"):
            TrainingConfig(learning_rate=2.0)
        
        # 测试无效的lora_rank
        with pytest.raises(ValueError, match="lora_rank必须大于0"):
            TrainingConfig(lora_rank=0)
    
    def test_training_config_auto_adjustment(self):
        """测试训练配置自动调整"""
        config = TrainingConfig(
            batch_size=8,
            per_device_batch_size=2,
            gradient_accumulation_steps=1  # 这个会被自动调整
        )
        
        # gradient_accumulation_steps应该被调整为4 (8/2)
        assert config.gradient_accumulation_steps == 4
    
    def test_training_config_serialization(self):
        """测试训练配置序列化"""
        config = TrainingConfig(
            num_epochs=3,
            learning_rate=2e-4,
            lora_target_modules=["q_proj", "v_proj"]
        )
        
        data = config.to_dict()
        assert data["num_epochs"] == 3
        assert data["learning_rate"] == 2e-4
        assert data["lora_target_modules"] == ["q_proj", "v_proj"]


class TestDataConfig:
    """数据配置测试"""
    
    def test_data_config_creation(self):
        """测试数据配置创建"""
        config = DataConfig(
            train_data_path="data/train",
            val_data_path="data/val",
            data_format="json",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        assert config.train_data_path == "data/train"
        assert config.data_format == "json"
        assert config.train_ratio == 0.8
    
    def test_data_config_validation(self):
        """测试数据配置验证"""
        # 测试比例和不等于1.0
        with pytest.raises(ValueError, match="训练、验证、测试集比例之和必须等于1.0"):
            DataConfig(
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.3  # 总和为1.1
            )
        
        # 测试无效的数据格式
        with pytest.raises(ValueError, match="data_format必须是'json'、'jsonl'或'csv'"):
            DataConfig(data_format="xml")
    
    def test_data_config_serialization(self):
        """测试数据配置序列化"""
        config = DataConfig(
            train_data_path="data/train",
            enable_chinese_processing=True,
            crypto_vocab_path="data/crypto.json"
        )
        
        data = config.to_dict()
        assert data["train_data_path"] == "data/train"
        assert data["enable_chinese_processing"] is True
        assert data["crypto_vocab_path"] == "data/crypto.json"


class TestSystemConfig:
    """系统配置测试"""
    
    def test_system_config_creation(self):
        """测试系统配置创建"""
        config = SystemConfig(
            parallel_strategy=ParallelStrategy.DATA_PARALLEL,
            data_parallel_size=2,
            output_dir="output",
            log_level=LogLevel.INFO
        )
        
        assert config.parallel_strategy == ParallelStrategy.DATA_PARALLEL
        assert config.data_parallel_size == 2
        assert config.output_dir == "output"
        assert config.log_level == LogLevel.INFO
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_system_config_auto_gpu_detection(self, mock_device_count, mock_cuda_available):
        """测试自动GPU检测"""
        config = SystemConfig(enable_multi_gpu=True)
        
        assert config.gpu_ids == [0, 1]
        assert config.num_gpus == 2
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_system_config_no_gpu(self, mock_cuda_available):
        """测试无GPU环境"""
        config = SystemConfig(enable_multi_gpu=True)
        
        assert config.enable_multi_gpu is False
        assert config.num_gpus == 1
    
    def test_system_config_single_gpu_adjustment(self):
        """测试单GPU配置调整"""
        config = SystemConfig(
            enable_multi_gpu=True,
            gpu_ids=[0],  # 只有一个GPU
            data_parallel_size=2
        )
        
        assert config.enable_multi_gpu is False
        assert config.data_parallel_size == 1
    
    def test_world_size_calculation(self):
        """测试world_size计算"""
        config = SystemConfig(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1
        )
        
        assert config.world_size == 4  # 2 * 2 * 1
    
    def test_device_map_generation(self):
        """测试设备映射生成"""
        # 单GPU情况
        config = SystemConfig(enable_multi_gpu=False)
        device_map = config.get_device_map()
        assert device_map == {"": 0}
        
        # 多GPU情况
        config = SystemConfig(
            enable_multi_gpu=True,
            gpu_ids=[0, 1, 2]
        )
        device_map = config.get_device_map()
        expected = {"gpu_0": 0, "gpu_1": 1, "gpu_2": 2}
        assert device_map == expected
    
    def test_system_config_serialization(self):
        """测试系统配置序列化"""
        config = SystemConfig(
            parallel_strategy=ParallelStrategy.HYBRID_PARALLEL,
            communication_backend=CommunicationBackend.NCCL,
            zero_stage=ZeroStage.OPTIMIZER_GRADIENT,
            log_level=LogLevel.DEBUG
        )
        
        data = config.to_dict()
        assert data["parallel_strategy"] == "hybrid_parallel"
        assert data["communication_backend"] == "nccl"
        assert data["zero_stage"] == 2
        assert data["log_level"] == "DEBUG"
        
        # 测试反序列化
        restored_config = SystemConfig.from_dict(data)
        assert restored_config.parallel_strategy == ParallelStrategy.HYBRID_PARALLEL
        assert restored_config.communication_backend == CommunicationBackend.NCCL
        assert restored_config.zero_stage == ZeroStage.OPTIMIZER_GRADIENT
        assert restored_config.log_level == LogLevel.DEBUG


class TestConfigManager:
    """配置管理器测试"""
    
    def test_config_manager_creation(self):
        """测试配置管理器创建"""
        manager = ConfigManager()
        assert manager.config is None
        
        manager_with_path = ConfigManager("config.yaml")
        assert manager_with_path.config_path == "config.yaml"
    
    def test_load_default_config(self):
        """测试加载默认配置"""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, SystemConfig)
        assert manager.config is not None
    
    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        manager = ConfigManager()
        config = manager.load_config("nonexistent.yaml")
        
        assert isinstance(config, SystemConfig)  # 应该返回默认配置
    
    def test_save_and_load_yaml_config(self):
        """测试保存和加载YAML配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # 创建测试配置
            original_config = SystemConfig(
                output_dir="test_output",
                log_level=LogLevel.DEBUG,
                data_parallel_size=2
            )
            
            # 保存配置
            manager = ConfigManager()
            success = manager.save_config(original_config, str(config_path), ConfigFormat.YAML)
            assert success is True
            assert config_path.exists()
            
            # 加载配置
            loaded_config = manager.load_config(str(config_path))
            assert loaded_config.output_dir == "test_output"
            assert loaded_config.log_level == LogLevel.DEBUG
            assert loaded_config.data_parallel_size == 2
    
    def test_save_and_load_json_config(self):
        """测试保存和加载JSON配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # 创建测试配置
            original_config = SystemConfig(
                cache_dir="test_cache",
                enable_monitoring=False
            )
            
            # 保存配置
            manager = ConfigManager()
            success = manager.save_config(original_config, str(config_path), ConfigFormat.JSON)
            assert success is True
            assert config_path.exists()
            
            # 加载配置
            loaded_config = manager.load_config(str(config_path))
            assert loaded_config.cache_dir == "test_cache"
            assert loaded_config.enable_monitoring is False
    
    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = SystemConfig(
            data_parallel_size=1,
            gpu_ids=[0],
            enable_multi_gpu=False
        )
        
        manager = ConfigManager()
        result = manager.validate_config(config)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_invalid_config(self):
        """测试验证无效配置"""
        # 创建一个配置，手动设置无效的并行度（绕过自动调整）
        config = SystemConfig()
        config.data_parallel_size = 4
        config.tensor_parallel_size = 2
        config.pipeline_parallel_size = 1
        config.gpu_ids = [0]  # 只有1个GPU但需要8个并行 (4*2*1)
        config.enable_multi_gpu = True
        
        manager = ConfigManager()
        result = manager.validate_config(config)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_create_template_config(self):
        """测试创建配置模板"""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "template.yaml"
            
            manager = ConfigManager()
            success = manager.create_template_config(str(template_path))
            
            assert success is True
            assert template_path.exists()
            
            # 验证模板内容
            with open(template_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            assert "_template_info" in data
            assert "description" in data["_template_info"]
    
    def test_merge_configs(self):
        """测试配置合并"""
        base_config = SystemConfig(
            output_dir="base_output",
            data_parallel_size=1,
            log_level=LogLevel.INFO
        )
        
        override_config = {
            "output_dir": "override_output",
            "data_parallel_size": 2,
            "training": {
                "learning_rate": 1e-5
            }
        }
        
        manager = ConfigManager()
        merged_config = manager.merge_configs(base_config, override_config)
        
        assert merged_config.output_dir == "override_output"
        assert merged_config.data_parallel_size == 2
        assert merged_config.log_level == LogLevel.INFO  # 保持原值
        assert merged_config.training.learning_rate == 1e-5
    
    def test_update_config(self):
        """测试更新配置"""
        manager = ConfigManager()
        config = manager.load_config()  # 加载默认配置
        
        updates = {
            "output_dir": "updated_output",
            "training": {
                "num_epochs": 10
            }
        }
        
        success = manager.update_config(updates)
        assert success is True
        
        updated_config = manager.get_config()
        assert updated_config.output_dir == "updated_output"
        assert updated_config.training.num_epochs == 10
    
    def test_update_config_without_config(self):
        """测试在没有配置时更新配置"""
        manager = ConfigManager()
        # 不加载配置
        
        success = manager.update_config({"output_dir": "test"})
        assert success is False
    
    def test_save_config_without_config(self):
        """测试在没有配置时保存配置"""
        manager = ConfigManager()
        # 不设置配置
        
        success = manager.save_config(config_path="test.yaml")
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])