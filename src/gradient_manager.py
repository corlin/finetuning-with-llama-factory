"""
梯度检查点和累积管理器

本模块实现了梯度检查点保存和恢复、梯度累积和内存优化策略、
激活值重计算机制和混合精度训练内存优化功能。
专门针对Qwen3-4B-Thinking模型的训练优化。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast
import logging
import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import gc
import warnings
from contextlib import contextmanager

from .memory_manager import MemoryManager, MemorySnapshot


@dataclass
class GradientCheckpointConfig:
    """梯度检查点配置"""
    enabled: bool = True
    checkpoint_layers: List[str] = field(default_factory=lambda: ["attention", "mlp"])
    checkpoint_ratio: float = 0.5  # 检查点层的比例
    preserve_rng_state: bool = True
    use_reentrant: bool = False  # PyTorch 2.0+ 推荐设为False
    
    # 内存优化配置
    offload_to_cpu: bool = False
    max_memory_usage: float = 0.8  # 最大内存使用率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "checkpoint_layers": self.checkpoint_layers,
            "checkpoint_ratio": self.checkpoint_ratio,
            "preserve_rng_state": self.preserve_rng_state,
            "use_reentrant": self.use_reentrant,
            "offload_to_cpu": self.offload_to_cpu,
            "max_memory_usage": self.max_memory_usage
        }


@dataclass
class GradientAccumulationConfig:
    """梯度累积配置"""
    enabled: bool = True
    accumulation_steps: int = 4
    sync_gradients: bool = True  # 是否在累积步骤间同步梯度
    
    # 动态调整配置
    adaptive_accumulation: bool = True
    min_accumulation_steps: int = 1
    max_accumulation_steps: int = 32
    memory_threshold: float = 0.85  # 内存使用率阈值
    
    # 梯度裁剪配置
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "accumulation_steps": self.accumulation_steps,
            "sync_gradients": self.sync_gradients,
            "adaptive_accumulation": self.adaptive_accumulation,
            "min_accumulation_steps": self.min_accumulation_steps,
            "max_accumulation_steps": self.max_accumulation_steps,
            "memory_threshold": self.memory_threshold,
            "gradient_clipping": self.gradient_clipping,
            "max_grad_norm": self.max_grad_norm
        }


@dataclass
class MixedPrecisionConfig:
    """混合精度配置"""
    enabled: bool = True
    dtype: str = "bfloat16"  # "float16" or "bfloat16"
    
    # GradScaler配置
    init_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # 自动损失缩放
    enabled_auto_scaling: bool = True
    
    # 内存优化
    cache_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "dtype": self.dtype,
            "init_scale": self.init_scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "enabled_auto_scaling": self.enabled_auto_scaling,
            "cache_enabled": self.cache_enabled
        }


@dataclass
class GradientStatistics:
    """梯度统计信息"""
    step: int
    total_norm: float
    max_grad: float
    min_grad: float
    num_zero_grads: int
    num_inf_grads: int
    num_nan_grads: int
    memory_usage: int  # MB
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step": self.step,
            "total_norm": self.total_norm,
            "max_grad": self.max_grad,
            "min_grad": self.min_grad,
            "num_zero_grads": self.num_zero_grads,
            "num_inf_grads": self.num_inf_grads,
            "num_nan_grads": self.num_nan_grads,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp.isoformat()
        }


class ActivationCheckpointing:
    """激活值检查点管理"""
    
    def __init__(self, config: GradientCheckpointConfig):
        """初始化激活值检查点管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.checkpointed_modules: Dict[str, nn.Module] = {}
        self.original_forwards: Dict[str, Callable] = {}
        
    def enable_checkpointing(self, model: nn.Module) -> bool:
        """为模型启用梯度检查点"""
        if not self.config.enabled:
            return False
        
        try:
            self._apply_checkpointing_to_model(model)
            self.logger.info("梯度检查点已启用")
            return True
        except Exception as e:
            self.logger.error(f"启用梯度检查点失败: {e}")
            return False
    
    def disable_checkpointing(self, model: nn.Module) -> bool:
        """禁用模型的梯度检查点"""
        try:
            self._restore_original_forwards(model)
            self.logger.info("梯度检查点已禁用")
            return True
        except Exception as e:
            self.logger.error(f"禁用梯度检查点失败: {e}")
            return False
    
    def _apply_checkpointing_to_model(self, model: nn.Module):
        """为模型应用检查点"""
        checkpoint_count = 0
        total_layers = 0
        
        for name, module in model.named_modules():
            total_layers += 1
            
            # 检查是否应该对该层应用检查点
            if self._should_checkpoint_layer(name, module):
                self._apply_checkpointing_to_layer(name, module)
                checkpoint_count += 1
        
        actual_ratio = checkpoint_count / total_layers if total_layers > 0 else 0
        self.logger.info(f"已对 {checkpoint_count}/{total_layers} 层应用检查点 "
                        f"(比例: {actual_ratio:.2%})")
    
    def _should_checkpoint_layer(self, name: str, module: nn.Module) -> bool:
        """判断是否应该对该层应用检查点"""
        # 检查层名称是否匹配配置
        for layer_type in self.config.checkpoint_layers:
            if layer_type.lower() in name.lower():
                return True
        
        # 检查模块类型
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, 
                              nn.TransformerDecoderLayer)):
            return True
        
        # 对于大型线性层也应用检查点
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            param_count = module.weight.numel()
            if param_count > 1000000:  # 超过100万参数的层
                return True
        
        return False
    
    def _apply_checkpointing_to_layer(self, name: str, module: nn.Module):
        """对特定层应用检查点"""
        if name in self.checkpointed_modules:
            return  # 已经应用过检查点
        
        # 保存原始forward方法
        original_forward = module.forward
        self.original_forwards[name] = original_forward
        
        # 创建检查点包装的forward方法
        def checkpointed_forward(*args, **kwargs):
            if module.training:
                # 训练模式下使用检查点
                return checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=self.config.use_reentrant,
                    preserve_rng_state=self.config.preserve_rng_state,
                    **kwargs
                )
            else:
                # 推理模式下直接调用
                return original_forward(*args, **kwargs)
        
        # 替换forward方法
        module.forward = checkpointed_forward
        self.checkpointed_modules[name] = module
    
    def _restore_original_forwards(self, model: nn.Module):
        """恢复原始的forward方法"""
        for name, original_forward in self.original_forwards.items():
            if name in self.checkpointed_modules:
                module = self.checkpointed_modules[name]
                module.forward = original_forward
        
        self.checkpointed_modules.clear()
        self.original_forwards.clear()
    
    @contextmanager
    def temporary_checkpointing(self, model: nn.Module):
        """临时启用检查点的上下文管理器"""
        was_enabled = len(self.checkpointed_modules) > 0
        
        if not was_enabled:
            self.enable_checkpointing(model)
        
        try:
            yield
        finally:
            if not was_enabled:
                self.disable_checkpointing(model)


class GradientAccumulator:
    """梯度累积管理器"""
    
    def __init__(self, config: GradientAccumulationConfig, memory_manager: Optional[MemoryManager] = None):
        """初始化梯度累积管理器"""
        self.config = config
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # 累积状态
        self.current_step = 0
        self.accumulated_steps = 0
        self.current_accumulation_steps = config.accumulation_steps
        
        # 梯度统计
        self.gradient_stats: List[GradientStatistics] = []
        self.max_stats_history = 1000
        
        # 分布式训练支持
        self.is_distributed = dist.is_available() and dist.is_initialized()
        
    def should_accumulate_gradients(self) -> bool:
        """判断是否应该累积梯度"""
        return self.accumulated_steps < self.current_accumulation_steps - 1
    
    def should_sync_gradients(self) -> bool:
        """判断是否应该同步梯度"""
        if not self.config.sync_gradients:
            return False
        
        return self.accumulated_steps == self.current_accumulation_steps - 1
    
    def accumulate_gradients(self, loss: torch.Tensor, model: nn.Module, 
                           scaler: Optional[GradScaler] = None) -> bool:
        """累积梯度"""
        try:
            # 缩放损失
            scaled_loss = loss / self.current_accumulation_steps
            
            # 反向传播
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            self.accumulated_steps += 1
            
            # 检查是否需要调整累积步数
            if self.config.adaptive_accumulation and self.memory_manager:
                self._adjust_accumulation_steps()
            
            return True
            
        except Exception as e:
            self.logger.error(f"梯度累积失败: {e}")
            return False
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer, 
                      scaler: Optional[GradScaler] = None, 
                      model: Optional[nn.Module] = None) -> bool:
        """执行优化器步骤"""
        if not self.should_sync_gradients():
            return False
        
        try:
            # 收集梯度统计（在优化器步骤之前）
            if model is not None:
                self._collect_gradient_statistics(model)
            
            # 梯度裁剪
            if self.config.gradient_clipping:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    self.config.max_grad_norm
                )
            
            # 优化器步骤
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 重置累积计数
            self.accumulated_steps = 0
            self.current_step += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"优化器步骤执行失败: {e}")
            return False
    
    def _adjust_accumulation_steps(self):
        """动态调整累积步数"""
        if not self.memory_manager:
            return
        
        memory_status = self.memory_manager.get_current_memory_status()
        if not memory_status:
            return
        
        # 根据内存使用率调整累积步数
        if memory_status.utilization_rate > self.config.memory_threshold:
            # 内存使用率过高，增加累积步数
            new_steps = min(
                self.config.max_accumulation_steps,
                int(self.current_accumulation_steps * 1.5)
            )
            if new_steps != self.current_accumulation_steps:
                self.logger.info(f"内存压力过高，增加梯度累积步数: "
                               f"{self.current_accumulation_steps} -> {new_steps}")
                self.current_accumulation_steps = new_steps
        
        elif memory_status.utilization_rate < self.config.memory_threshold * 0.7:
            # 内存使用率较低，可以减少累积步数
            new_steps = max(
                self.config.min_accumulation_steps,
                int(self.current_accumulation_steps * 0.8)
            )
            if new_steps != self.current_accumulation_steps:
                self.logger.info(f"内存使用率较低，减少梯度累积步数: "
                               f"{self.current_accumulation_steps} -> {new_steps}")
                self.current_accumulation_steps = new_steps
    
    def _collect_gradient_statistics(self, model: nn.Module):
        """收集梯度统计信息"""
        try:
            total_norm = 0.0
            max_grad = 0.0
            min_grad = float('inf')
            num_zero_grads = 0
            num_inf_grads = 0
            num_nan_grads = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_data = param.grad.data
                    
                    # 计算范数
                    param_norm = grad_data.norm().item()
                    total_norm += param_norm ** 2
                    
                    # 统计最大最小值
                    param_max = grad_data.max().item()
                    param_min = grad_data.min().item()
                    max_grad = max(max_grad, param_max)
                    min_grad = min(min_grad, param_min)
                    
                    # 统计特殊值
                    num_zero_grads += (grad_data == 0).sum().item()
                    num_inf_grads += torch.isinf(grad_data).sum().item()
                    num_nan_grads += torch.isnan(grad_data).sum().item()
            
            total_norm = total_norm ** 0.5
            
            # 获取内存使用情况
            memory_usage = 0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() // (1024**2)
            
            # 创建统计记录
            stats = GradientStatistics(
                step=self.current_step,
                total_norm=total_norm,
                max_grad=max_grad,
                min_grad=min_grad if min_grad != float('inf') else 0.0,
                num_zero_grads=num_zero_grads,
                num_inf_grads=num_inf_grads,
                num_nan_grads=num_nan_grads,
                memory_usage=memory_usage
            )
            
            self.gradient_stats.append(stats)
            
            # 限制历史记录长度
            if len(self.gradient_stats) > self.max_stats_history:
                self.gradient_stats = self.gradient_stats[-self.max_stats_history:]
            
            # 检查梯度异常
            if num_inf_grads > 0 or num_nan_grads > 0:
                self.logger.warning(f"检测到异常梯度: inf={num_inf_grads}, nan={num_nan_grads}")
            
        except Exception as e:
            self.logger.error(f"收集梯度统计失败: {e}")
    
    def get_gradient_statistics(self, last_n_steps: int = 100) -> List[GradientStatistics]:
        """获取梯度统计信息"""
        return self.gradient_stats[-last_n_steps:]
    
    def reset_accumulation(self):
        """重置累积状态"""
        self.accumulated_steps = 0
        self.current_step = 0


class MixedPrecisionManager:
    """混合精度训练管理器"""
    
    def __init__(self, config: MixedPrecisionConfig):
        """初始化混合精度管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化GradScaler
        self.scaler = None
        if self.config.enabled and self.config.enabled_auto_scaling:
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=self.config.enabled
            )
        
        # 设置数据类型
        self.dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        
        # 缓存管理
        if self.config.cache_enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @contextmanager
    def autocast_context(self):
        """自动混合精度上下文管理器"""
        if self.config.enabled:
            with autocast(dtype=self.dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: torch.Tensor):
        """反向传播"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """执行优化器步骤"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """取消梯度缩放"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)


class GradientManager:
    """梯度管理器主类"""
    
    def __init__(self, 
                 checkpoint_config: Optional[GradientCheckpointConfig] = None,
                 accumulation_config: Optional[GradientAccumulationConfig] = None,
                 mixed_precision_config: Optional[MixedPrecisionConfig] = None,
                 memory_manager: Optional[MemoryManager] = None):
        """初始化梯度管理器"""
        
        self.logger = logging.getLogger(__name__)
        
        # 配置初始化
        self.checkpoint_config = checkpoint_config or GradientCheckpointConfig()
        self.accumulation_config = accumulation_config or GradientAccumulationConfig()
        self.mixed_precision_config = mixed_precision_config or MixedPrecisionConfig()
        
        # 组件初始化
        self.activation_checkpointing = ActivationCheckpointing(self.checkpoint_config)
        self.gradient_accumulator = GradientAccumulator(self.accumulation_config, memory_manager)
        self.mixed_precision_manager = MixedPrecisionManager(self.mixed_precision_config)
        
        # 状态管理
        self.is_initialized = False
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
    def initialize(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """初始化梯度管理器"""
        try:
            self.model = model
            self.optimizer = optimizer
            
            # 启用梯度检查点
            if self.checkpoint_config.enabled:
                success = self.activation_checkpointing.enable_checkpointing(model)
                if not success:
                    self.logger.warning("梯度检查点启用失败")
            
            self.is_initialized = True
            self.logger.info("梯度管理器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"梯度管理器初始化失败: {e}")
            return False
    
    def training_step(self, loss: torch.Tensor) -> bool:
        """执行训练步骤"""
        if not self.is_initialized:
            self.logger.error("梯度管理器未初始化")
            return False
        
        try:
            # 累积梯度
            success = self.gradient_accumulator.accumulate_gradients(
                loss, self.model, self.mixed_precision_manager.scaler
            )
            
            if not success:
                return False
            
            # 检查是否需要执行优化器步骤
            if self.gradient_accumulator.should_sync_gradients():
                success = self.gradient_accumulator.step_optimizer(
                    self.optimizer, self.mixed_precision_manager.scaler, self.model
                )
                return success
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练步骤执行失败: {e}")
            return False
    
    @contextmanager
    def autocast_context(self):
        """混合精度上下文管理器"""
        with self.mixed_precision_manager.autocast_context():
            yield
    
    def get_gradient_statistics(self, last_n_steps: int = 100) -> List[GradientStatistics]:
        """获取梯度统计信息"""
        return self.gradient_accumulator.get_gradient_statistics(last_n_steps)
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """保存检查点"""
        try:
            checkpoint_data = {
                "gradient_accumulator_step": self.gradient_accumulator.current_step,
                "accumulated_steps": self.gradient_accumulator.accumulated_steps,
                "current_accumulation_steps": self.gradient_accumulator.current_accumulation_steps,
                "mixed_precision_scaler": self.mixed_precision_manager.state_dict(),
                "checkpoint_config": self.checkpoint_config.to_dict(),
                "accumulation_config": self.accumulation_config.to_dict(),
                "mixed_precision_config": self.mixed_precision_config.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"梯度管理器检查点已保存: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
                return False
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 恢复状态
            self.gradient_accumulator.current_step = checkpoint_data.get("gradient_accumulator_step", 0)
            self.gradient_accumulator.accumulated_steps = checkpoint_data.get("accumulated_steps", 0)
            self.gradient_accumulator.current_accumulation_steps = checkpoint_data.get("current_accumulation_steps", 4)
            
            # 恢复混合精度状态
            scaler_state = checkpoint_data.get("mixed_precision_scaler", {})
            if scaler_state:
                self.mixed_precision_manager.load_state_dict(scaler_state)
            
            self.logger.info(f"梯度管理器检查点已加载: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        optimizations = {
            "gradient_checkpointing": False,
            "increased_accumulation": False,
            "cache_cleared": False,
            "memory_saved_mb": 0
        }
        
        try:
            # 获取当前内存使用
            initial_memory = 0
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() // (1024**2)
            
            # 启用梯度检查点（如果未启用）
            if not self.checkpoint_config.enabled and self.model:
                self.checkpoint_config.enabled = True
                self.activation_checkpointing.config = self.checkpoint_config
                success = self.activation_checkpointing.enable_checkpointing(self.model)
                if success:
                    optimizations["gradient_checkpointing"] = True
            
            # 增加梯度累积步数
            if self.gradient_accumulator.current_accumulation_steps < self.accumulation_config.max_accumulation_steps:
                old_steps = self.gradient_accumulator.current_accumulation_steps
                new_steps = min(
                    self.accumulation_config.max_accumulation_steps,
                    old_steps * 2
                )
                self.gradient_accumulator.current_accumulation_steps = new_steps
                optimizations["increased_accumulation"] = True
                self.logger.info(f"增加梯度累积步数: {old_steps} -> {new_steps}")
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimizations["cache_cleared"] = True
            
            gc.collect()
            
            # 计算节省的内存
            final_memory = 0
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() // (1024**2)
            
            optimizations["memory_saved_mb"] = max(0, initial_memory - final_memory)
            
            self.logger.info(f"内存优化完成，节省内存: {optimizations['memory_saved_mb']}MB")
            
        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
        
        return optimizations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {
            "gradient_accumulation_steps": self.gradient_accumulator.current_accumulation_steps,
            "current_step": self.gradient_accumulator.current_step,
            "accumulated_steps": self.gradient_accumulator.accumulated_steps,
            "gradient_checkpointing_enabled": self.checkpoint_config.enabled,
            "mixed_precision_enabled": self.mixed_precision_config.enabled,
            "mixed_precision_scale": self.mixed_precision_manager.get_scale(),
            "checkpointed_modules": len(self.activation_checkpointing.checkpointed_modules)
        }
        
        # 添加最近的梯度统计
        recent_stats = self.get_gradient_statistics(10)
        if recent_stats:
            latest_stats = recent_stats[-1]
            metrics.update({
                "latest_gradient_norm": latest_stats.total_norm,
                "latest_memory_usage": latest_stats.memory_usage,
                "gradient_anomalies": {
                    "inf_grads": latest_stats.num_inf_grads,
                    "nan_grads": latest_stats.num_nan_grads,
                    "zero_grads": latest_stats.num_zero_grads
                }
            })
        
        return metrics
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.model and self.activation_checkpointing:
                self.activation_checkpointing.disable_checkpointing(self.model)
            
            if self.gradient_accumulator:
                self.gradient_accumulator.reset_accumulation()
            
            self.is_initialized = False
            self.logger.info("梯度管理器资源已清理")
            
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()