#!/usr/bin/env python3
"""
训练流水线编排器测试
"""

import sys
import os
import tempfile
import shutil
import time
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training_pipeline import (
    TrainingPipelineOrchestrator, PipelineStage, PipelineStatus
)
from data_models import TrainingExample, DifficultyLevel
from config_manager import TrainingConfig, DataConfig
from lora_config_optimizer import LoRAMemoryProfile
from parallel_config import ParallelConfig, ParallelStrategy


def test_pipeline_basic_functionality():
    """测试流水线基本功能"""
    print("测试训练流水线编排器基本功能...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    try:
        # 创建测试数据
        test_data = [
            TrainingExample(
                instruction="什么是AES加密？",
                input="",
                output="AES是高级加密标准。",
                thinking="<thinking>需要解释AES的基本概念。</thinking>",
                crypto_terms=["AES", "加密"],
                difficulty_level=DifficultyLevel.BEGINNER
            ),
            TrainingExample(
                instruction="比较RSA和ECC算法",
                input="在现代密码学中",
                output="RSA基于大整数分解，ECC基于椭圆曲线。",
                crypto_terms=["RSA", "ECC"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
        ]
        
        # 创建配置
        training_config = TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            output_dir=str(Path(temp_dir) / "model_output")
        )
        data_config = DataConfig()
        lora_config = LoRAMemoryProfile(
            rank=8, alpha=16, target_modules=["q_proj", "v_proj"]
        )
        parallel_config = ParallelConfig(strategy=ParallelStrategy.DATA_PARALLEL)
        
        # 创建流水线编排器
        orchestrator = TrainingPipelineOrchestrator(
            "test_pipeline",
            output_dir=temp_dir
        )
        print("✓ 流水线编排器创建成功")
        
        # 配置流水线
        orchestrator.configure_pipeline(
            test_data, training_config, data_config, lora_config, parallel_config
        )
        print("✓ 流水线配置完成")
        
        # 添加进度回调
        progress_updates = []
        def progress_callback(state):
            progress_updates.append((state.current_stage, state.progress))
            print(f"  进度更新: {state.current_stage.value} - {state.progress:.1f}%")
        
        orchestrator.add_progress_callback(progress_callback)
        
        # 添加阶段回调
        stage_completions = []
        def stage_callback(state):
            stage_completions.append(state.current_stage)
            print(f"  阶段完成: {state.current_stage.value}")
        
        for stage in [PipelineStage.INITIALIZATION, PipelineStage.DATA_PREPARATION, 
                     PipelineStage.CONFIG_GENERATION]:
            orchestrator.add_stage_callback(stage, stage_callback)
        
        # 测试流水线状态
        assert orchestrator.get_state().status == PipelineStatus.PENDING
        assert not orchestrator.is_running()
        assert not orchestrator.is_completed()
        print("✓ 初始状态验证通过")
        
        # 运行流水线（只运行前几个阶段，避免实际训练）
        print("开始运行流水线...")
        
        # 手动执行前几个阶段进行测试
        success = True
        
        # 测试初始化阶段
        orchestrator.state.status = PipelineStatus.RUNNING
        orchestrator.state.start_time = orchestrator.state.start_time or time.time()
        
        if orchestrator._stage_initialization():
            print("✓ 初始化阶段完成")
        else:
            print("✗ 初始化阶段失败")
            success = False
        
        # 测试数据准备阶段
        if success and orchestrator._stage_data_preparation():
            print("✓ 数据准备阶段完成")
            print(f"  数据文件: {orchestrator.data_files}")
        else:
            print("✗ 数据准备阶段失败")
            success = False
        
        # 测试配置生成阶段
        if success and orchestrator._stage_config_generation():
            print("✓ 配置生成阶段完成")
            print(f"  配置文件: {orchestrator.config_files}")
        else:
            print("✗ 配置生成阶段失败")
            success = False
        
        # 验证输出文件
        if success:
            output_dir = Path(temp_dir)
            expected_dirs = ["data", "configs", "checkpoints", "logs", "models"]
            for dir_name in expected_dirs:
                if (output_dir / dir_name).exists():
                    print(f"  ✓ {dir_name} 目录存在")
                else:
                    print(f"  ✗ {dir_name} 目录不存在")
            
            # 检查数据文件
            if orchestrator.data_files:
                for file_type, file_path in orchestrator.data_files.items():
                    if Path(file_path).exists():
                        print(f"  ✓ {file_type} 文件存在")
                    else:
                        print(f"  ✗ {file_type} 文件不存在")
            
            # 检查配置文件
            if orchestrator.config_files:
                for file_type, file_path in orchestrator.config_files.items():
                    if Path(file_path).exists():
                        print(f"  ✓ {file_type} 文件存在")
                    else:
                        print(f"  ✗ {file_type} 文件不存在")
        
        # 测试状态管理
        state = orchestrator.get_state()
        print(f"✓ 流水线状态: {state.status.value}")
        print(f"✓ 当前阶段: {state.current_stage.value}")
        print(f"✓ 进度: {state.progress:.1f}%")
        
        # 测试检查点功能
        orchestrator._create_checkpoint(PipelineStage.DATA_PREPARATION)
        if state.checkpoints:
            print("✓ 检查点创建成功")
        else:
            print("✗ 检查点创建失败")
        
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"清理临时目录: {temp_dir}")


def test_pipeline_control():
    """测试流水线控制功能"""
    print("\n测试流水线控制功能...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建简单的测试数据
        test_data = [
            TrainingExample(
                instruction="测试问题",
                input="",
                output="测试回答",
                crypto_terms=["测试"]
            )
        ]
        
        # 创建配置
        training_config = TrainingConfig(num_train_epochs=1)
        data_config = DataConfig()
        lora_config = LoRAMemoryProfile(rank=4, alpha=8, target_modules=["q_proj"])
        parallel_config = ParallelConfig(strategy=ParallelStrategy.DATA_PARALLEL)
        
        # 创建流水线编排器
        orchestrator = TrainingPipelineOrchestrator("control_test", temp_dir)
        orchestrator.configure_pipeline(
            test_data, training_config, data_config, lora_config, parallel_config
        )
        
        # 测试暂停和恢复
        orchestrator.pause_pipeline()
        assert orchestrator.should_pause
        print("✓ 暂停功能测试通过")
        
        orchestrator.resume_pipeline()
        assert not orchestrator.should_pause
        print("✓ 恢复功能测试通过")
        
        # 测试停止
        orchestrator.stop_pipeline()
        assert orchestrator.should_stop
        print("✓ 停止功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 控制功能测试失败: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir)


def test_pipeline_state_management():
    """测试流水线状态管理"""
    print("\n测试流水线状态管理...")
    
    try:
        from training_pipeline import PipelineState, PipelineCheckpoint
        
        # 创建流水线状态
        state = PipelineState("test_state")
        
        # 测试阶段更新
        state.update_stage(PipelineStage.INITIALIZATION, 50.0)
        assert state.current_stage == PipelineStage.INITIALIZATION
        assert state.stage_progress[PipelineStage.INITIALIZATION] == 50.0
        print("✓ 阶段更新测试通过")
        
        # 测试进度更新
        state.update_stage_progress(PipelineStage.INITIALIZATION, 100.0)
        assert state.stage_progress[PipelineStage.INITIALIZATION] == 100.0
        print("✓ 进度更新测试通过")
        
        # 测试检查点
        from datetime import datetime
        checkpoint = PipelineCheckpoint(
            checkpoint_id="test_checkpoint",
            stage=PipelineStage.INITIALIZATION,
            timestamp=datetime.now(),
            state_data={"test": "data"}
        )
        
        state.add_checkpoint(checkpoint)
        assert len(state.checkpoints) == 1
        assert state.latest_checkpoint == checkpoint
        print("✓ 检查点管理测试通过")
        
        # 测试序列化
        state_dict = state.to_dict()
        assert "pipeline_id" in state_dict
        assert "status" in state_dict
        assert "current_stage" in state_dict
        print("✓ 状态序列化测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 状态管理测试失败: {e}")
        return False


def test_integration_with_direct_training():
    """测试与直接训练引擎的集成"""
    print("\n测试与直接训练引擎的集成...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建更复杂的测试数据
        test_data = []
        for i in range(5):
            example = TrainingExample(
                instruction=f"密码学问题 {i+1}",
                input=f"上下文 {i+1}",
                output=f"这是关于密码学的回答 {i+1}",
                thinking=f"<thinking>这是思考过程 {i+1}</thinking>",
                crypto_terms=["密码学", "安全"],
                difficulty_level=DifficultyLevel.INTERMEDIATE
            )
            test_data.append(example)
        
        # 创建配置
        training_config = TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-4
        )
        data_config = DataConfig()
        lora_config = LoRAMemoryProfile(
            rank=8, alpha=16, target_modules=["q_proj", "v_proj"]
        )
        parallel_config = ParallelConfig(strategy=ParallelStrategy.DATA_PARALLEL)
        
        # 创建流水线编排器
        orchestrator = TrainingPipelineOrchestrator("integration_test", temp_dir)
        orchestrator.configure_pipeline(
            test_data, training_config, data_config, lora_config, parallel_config
        )
        
        # 执行数据准备和配置生成阶段
        success = (orchestrator._stage_initialization() and
                  orchestrator._stage_data_preparation() and
                  orchestrator._stage_config_generation())
        
        if success:
            print("✓ 直接训练引擎集成测试通过")
            
            # 验证生成的文件
            if orchestrator.data_files and orchestrator.config_files:
                print("  ✓ 数据文件和配置文件都已生成")
                
                # 检查直接训练配置文件内容
                config_file = orchestrator.config_files.get("direct_training_config")
                if config_file and Path(config_file).exists():
                    print("  ✓ 直接训练配置文件存在")
                    
                    # 读取配置文件验证内容
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    required_keys = ["model_name", "data_path", "lora_r", "lora_alpha"]
                    if all(key in config for key in required_keys):
                        print("  ✓ 配置文件包含必需字段")
                    else:
                        print("  ✗ 配置文件缺少必需字段")
                        success = False
                
                # 检查训练脚本
                script_file = orchestrator.config_files.get("training_script")
                if script_file and Path(script_file).exists():
                    print("  ✓ 训练脚本文件存在")
                else:
                    print("  ✗ 训练脚本文件不存在")
                    success = False
            else:
                print("  ✗ 文件生成不完整")
                success = False
        else:
            print("✗ 直接训练引擎集成测试失败")
        
        return success
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("训练流水线编排器测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_pipeline_basic_functionality())
    test_results.append(test_pipeline_control())
    test_results.append(test_pipeline_state_management())
    test_results.append(test_integration_with_direct_training())
    
    # 汇总结果
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
        exit(0)
    else:
        print("❌ 部分测试失败")
        exit(1)