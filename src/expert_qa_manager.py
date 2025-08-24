"""
专家QA数据集管理器

本模块负责管理专家QA数据集的构建、维护和质量控制，
包括专家标注工作流、数据集版本管理和评估一致性检查。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import hashlib
from datetime import datetime
import logging
from pathlib import Path

from .evaluation_framework import ExpertQAItem, EvaluationDimension, ExpertiseLevel


class AnnotationStatus(Enum):
    """标注状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExpertRole(Enum):
    """专家角色"""
    DOMAIN_EXPERT = "domain_expert"  # 领域专家
    LINGUISTIC_EXPERT = "linguistic_expert"  # 语言专家
    EVALUATION_EXPERT = "evaluation_expert"  # 评估专家
    SENIOR_REVIEWER = "senior_reviewer"  # 高级审核员


@dataclass
class ExpertProfile:
    """专家档案"""
    expert_id: str
    name: str
    role: ExpertRole
    expertise_areas: List[str]
    experience_years: int
    certification_level: int
    contact_info: Dict[str, str]
    annotation_history: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "name": self.name,
            "role": self.role.value,
            "expertise_areas": self.expertise_areas,
            "experience_years": self.experience_years,
            "certification_level": self.certification_level,
            "contact_info": self.contact_info,
            "annotation_history": self.annotation_history,
            "quality_score": self.quality_score
        }


@dataclass
class AnnotationTask:
    """标注任务"""
    task_id: str
    qa_item: ExpertQAItem
    assigned_experts: List[str]
    status: AnnotationStatus
    created_time: datetime
    deadline: datetime
    annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consensus_reached: bool = False
    final_annotation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "qa_item": self.qa_item.to_dict(),
            "assigned_experts": self.assigned_experts,
            "status": self.status.value,
            "created_time": self.created_time.isoformat(),
            "deadline": self.deadline.isoformat(),
            "annotations": self.annotations,
            "consensus_reached": self.consensus_reached,
            "final_annotation": self.final_annotation
        }


class ExpertQAManager:
    """专家QA数据集管理器"""
    
    def __init__(self, data_dir: str, logger: Optional[logging.Logger] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # 数据存储路径
        self.experts_file = self.data_dir / "experts.json"
        self.qa_items_file = self.data_dir / "qa_items.json"
        self.tasks_file = self.data_dir / "annotation_tasks.json"
        self.consensus_file = self.data_dir / "consensus_data.json"
        
        # 内存数据
        self.experts: Dict[str, ExpertProfile] = {}
        self.qa_items: Dict[str, ExpertQAItem] = {}
        self.annotation_tasks: Dict[str, AnnotationTask] = {}
        
        # 加载现有数据
        self._load_data()
    
    def register_expert(self, expert: ExpertProfile) -> bool:
        """
        注册专家
        
        Args:
            expert: 专家档案
            
        Returns:
            bool: 注册是否成功
        """
        try:
            if expert.expert_id in self.experts:
                self.logger.warning(f"专家 {expert.expert_id} 已存在")
                return False
            
            self.experts[expert.expert_id] = expert
            self._save_experts()
            
            self.logger.info(f"成功注册专家: {expert.name} ({expert.expert_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"注册专家失败: {e}")
            return False
    
    def add_qa_item(self, qa_item: ExpertQAItem) -> bool:
        """
        添加QA数据项
        
        Args:
            qa_item: QA数据项
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if qa_item.question_id in self.qa_items:
                self.logger.warning(f"QA项 {qa_item.question_id} 已存在")
                return False
            
            self.qa_items[qa_item.question_id] = qa_item
            self._save_qa_items()
            
            self.logger.info(f"成功添加QA项: {qa_item.question_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加QA项失败: {e}")
            return False
    
    def create_annotation_task(self, qa_item_id: str, expert_ids: List[str], 
                             deadline_days: int = 7) -> Optional[str]:
        """
        创建标注任务
        
        Args:
            qa_item_id: QA项ID
            expert_ids: 分配的专家ID列表
            deadline_days: 截止天数
            
        Returns:
            Optional[str]: 任务ID，失败时返回None
        """
        try:
            if qa_item_id not in self.qa_items:
                self.logger.error(f"QA项 {qa_item_id} 不存在")
                return None
            
            # 验证专家存在
            for expert_id in expert_ids:
                if expert_id not in self.experts:
                    self.logger.error(f"专家 {expert_id} 不存在")
                    return None
            
            # 生成任务ID
            task_id = self._generate_task_id(qa_item_id, expert_ids)
            
            # 创建任务
            task = AnnotationTask(
                task_id=task_id,
                qa_item=self.qa_items[qa_item_id],
                assigned_experts=expert_ids,
                status=AnnotationStatus.PENDING,
                created_time=datetime.now(),
                deadline=datetime.now().replace(day=datetime.now().day + deadline_days)
            )
            
            self.annotation_tasks[task_id] = task
            self._save_tasks()
            
            self.logger.info(f"成功创建标注任务: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"创建标注任务失败: {e}")
            return None
    
    def submit_annotation(self, task_id: str, expert_id: str, 
                         annotation: Dict[str, Any]) -> bool:
        """
        提交标注结果
        
        Args:
            task_id: 任务ID
            expert_id: 专家ID
            annotation: 标注结果
            
        Returns:
            bool: 提交是否成功
        """
        try:
            if task_id not in self.annotation_tasks:
                self.logger.error(f"标注任务 {task_id} 不存在")
                return False
            
            task = self.annotation_tasks[task_id]
            
            if expert_id not in task.assigned_experts:
                self.logger.error(f"专家 {expert_id} 未分配到任务 {task_id}")
                return False
            
            # 验证标注格式
            if not self._validate_annotation(annotation):
                self.logger.error("标注格式不正确")
                return False
            
            # 保存标注
            task.annotations[expert_id] = {
                "annotation": annotation,
                "timestamp": datetime.now().isoformat(),
                "expert_profile": self.experts[expert_id].to_dict()
            }
            
            # 更新任务状态
            if len(task.annotations) == len(task.assigned_experts):
                task.status = AnnotationStatus.COMPLETED
                # 尝试达成共识
                self._attempt_consensus(task_id)
            elif task.status == AnnotationStatus.PENDING:
                task.status = AnnotationStatus.IN_PROGRESS
            
            self._save_tasks()
            
            self.logger.info(f"专家 {expert_id} 成功提交任务 {task_id} 的标注")
            return True
            
        except Exception as e:
            self.logger.error(f"提交标注失败: {e}")
            return False
    
    def calculate_inter_annotator_agreement(self, task_id: str) -> Dict[str, float]:
        """
        计算标注者间一致性
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, float]: 各维度的一致性分数
        """
        try:
            if task_id not in self.annotation_tasks:
                return {}
            
            task = self.annotation_tasks[task_id]
            annotations = task.annotations
            
            if len(annotations) < 2:
                return {}
            
            agreement_scores = {}
            
            # 计算各评估维度的一致性
            for dimension in EvaluationDimension:
                scores = []
                for expert_id, annotation_data in annotations.items():
                    annotation = annotation_data["annotation"]
                    if dimension.value in annotation.get("scores", {}):
                        scores.append(annotation["scores"][dimension.value])
                
                if len(scores) >= 2:
                    # 计算标准差作为一致性指标（标准差越小，一致性越高）
                    import numpy as np
                    std_dev = np.std(scores)
                    # 转换为一致性分数（0-1，越高越一致）
                    agreement_scores[dimension.value] = max(0, 1 - std_dev)
            
            return agreement_scores
            
        except Exception as e:
            self.logger.error(f"计算标注者间一致性失败: {e}")
            return {}
    
    def _attempt_consensus(self, task_id: str) -> bool:
        """
        尝试达成标注共识
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否达成共识
        """
        try:
            task = self.annotation_tasks[task_id]
            annotations = task.annotations
            
            if len(annotations) < 2:
                return False
            
            # 计算一致性
            agreement_scores = self.calculate_inter_annotator_agreement(task_id)
            
            # 设定一致性阈值
            consensus_threshold = 0.8
            
            # 检查是否达成共识
            high_agreement_dimensions = [
                dim for dim, score in agreement_scores.items() 
                if score >= consensus_threshold
            ]
            
            if len(high_agreement_dimensions) >= len(agreement_scores) * 0.7:
                # 达成共识，计算最终标注
                final_annotation = self._calculate_consensus_annotation(annotations)
                task.final_annotation = final_annotation
                task.consensus_reached = True
                task.status = AnnotationStatus.APPROVED
                
                self.logger.info(f"任务 {task_id} 达成标注共识")
                return True
            else:
                # 未达成共识，需要进一步讨论或重新标注
                task.status = AnnotationStatus.REVIEWED
                self.logger.warning(f"任务 {task_id} 未达成标注共识，需要进一步处理")
                return False
                
        except Exception as e:
            self.logger.error(f"尝试达成共识失败: {e}")
            return False
    
    def _calculate_consensus_annotation(self, annotations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算共识标注
        
        Args:
            annotations: 所有标注结果
            
        Returns:
            Dict[str, Any]: 共识标注结果
        """
        import numpy as np
        
        consensus = {
            "scores": {},
            "feedback": {},
            "consensus_method": "weighted_average",
            "annotator_count": len(annotations)
        }
        
        # 计算各维度的加权平均分
        for dimension in EvaluationDimension:
            scores = []
            weights = []
            
            for expert_id, annotation_data in annotations.items():
                annotation = annotation_data["annotation"]
                expert = self.experts[expert_id]
                
                if dimension.value in annotation.get("scores", {}):
                    scores.append(annotation["scores"][dimension.value])
                    # 专家权重基于经验和质量分数
                    weight = expert.quality_score * (1 + expert.experience_years * 0.1)
                    weights.append(weight)
            
            if scores:
                # 加权平均
                weighted_avg = np.average(scores, weights=weights)
                consensus["scores"][dimension.value] = float(weighted_avg)
        
        # 合并反馈意见
        all_feedback = []
        for annotation_data in annotations.values():
            annotation = annotation_data["annotation"]
            if "feedback" in annotation:
                all_feedback.append(annotation["feedback"])
        
        consensus["feedback"]["combined"] = "; ".join(all_feedback)
        
        return consensus
    
    def get_quality_control_report(self) -> Dict[str, Any]:
        """
        生成质量控制报告
        
        Returns:
            Dict[str, Any]: 质量控制报告
        """
        try:
            completed_tasks = [
                task for task in self.annotation_tasks.values()
                if task.status == AnnotationStatus.COMPLETED or task.status == AnnotationStatus.APPROVED
            ]
            
            if not completed_tasks:
                return {"error": "没有已完成的标注任务"}
            
            # 统计专家表现
            expert_stats = {}
            for expert_id, expert in self.experts.items():
                expert_stats[expert_id] = {
                    "name": expert.name,
                    "role": expert.role.value,
                    "tasks_completed": 0,
                    "average_agreement": 0.0,
                    "quality_score": expert.quality_score
                }
            
            # 计算统计数据
            total_agreement_scores = []
            consensus_reached_count = 0
            
            for task in completed_tasks:
                if task.consensus_reached:
                    consensus_reached_count += 1
                
                # 更新专家统计
                for expert_id in task.assigned_experts:
                    if expert_id in expert_stats:
                        expert_stats[expert_id]["tasks_completed"] += 1
                
                # 计算一致性
                agreement_scores = self.calculate_inter_annotator_agreement(task.task_id)
                if agreement_scores:
                    avg_agreement = sum(agreement_scores.values()) / len(agreement_scores)
                    total_agreement_scores.append(avg_agreement)
            
            # 生成报告
            report = {
                "summary": {
                    "total_tasks": len(completed_tasks),
                    "consensus_reached": consensus_reached_count,
                    "consensus_rate": consensus_reached_count / len(completed_tasks) if completed_tasks else 0,
                    "average_inter_annotator_agreement": sum(total_agreement_scores) / len(total_agreement_scores) if total_agreement_scores else 0
                },
                "expert_performance": expert_stats,
                "recommendations": self._generate_quality_recommendations(expert_stats, total_agreement_scores),
                "timestamp": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成质量控制报告失败: {e}")
            return {"error": str(e)}
    
    def _generate_quality_recommendations(self, expert_stats: Dict[str, Any], 
                                        agreement_scores: List[float]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        if agreement_scores:
            avg_agreement = sum(agreement_scores) / len(agreement_scores)
            if avg_agreement < 0.7:
                recommendations.append("整体标注一致性较低，建议加强专家培训和标注指南")
        
        # 检查专家表现
        low_performing_experts = [
            expert_id for expert_id, stats in expert_stats.items()
            if stats["quality_score"] < 0.8
        ]
        
        if low_performing_experts:
            recommendations.append(f"以下专家需要额外培训: {', '.join(low_performing_experts)}")
        
        return recommendations
    
    def _generate_task_id(self, qa_item_id: str, expert_ids: List[str]) -> str:
        """生成任务ID"""
        content = f"{qa_item_id}_{sorted(expert_ids)}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        """验证标注格式"""
        required_fields = ["scores", "feedback"]
        return all(field in annotation for field in required_fields)
    
    def _load_data(self):
        """加载数据"""
        try:
            # 加载专家数据
            if self.experts_file.exists():
                with open(self.experts_file, 'r', encoding='utf-8') as f:
                    experts_data = json.load(f)
                    for expert_id, expert_dict in experts_data.items():
                        expert_dict["role"] = ExpertRole(expert_dict["role"])
                        self.experts[expert_id] = ExpertProfile(**expert_dict)
            
            # 加载QA数据
            if self.qa_items_file.exists():
                with open(self.qa_items_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                    for qa_id, qa_dict in qa_data.items():
                        qa_dict["difficulty_level"] = ExpertiseLevel(qa_dict["difficulty_level"])
                        qa_dict["evaluation_criteria"] = {
                            EvaluationDimension(k): v for k, v in qa_dict["evaluation_criteria"].items()
                        }
                        self.qa_items[qa_id] = ExpertQAItem(**qa_dict)
            
            # 加载任务数据
            if self.tasks_file.exists():
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                    for task_id, task_dict in tasks_data.items():
                        task_dict["status"] = AnnotationStatus(task_dict["status"])
                        task_dict["created_time"] = datetime.fromisoformat(task_dict["created_time"])
                        task_dict["deadline"] = datetime.fromisoformat(task_dict["deadline"])
                        
                        # 重建QA项
                        qa_dict = task_dict["qa_item"]
                        qa_dict["difficulty_level"] = ExpertiseLevel(qa_dict["difficulty_level"])
                        qa_dict["evaluation_criteria"] = {
                            EvaluationDimension(k): v for k, v in qa_dict["evaluation_criteria"].items()
                        }
                        task_dict["qa_item"] = ExpertQAItem(**qa_dict)
                        
                        self.annotation_tasks[task_id] = AnnotationTask(**task_dict)
                        
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
    
    def _save_experts(self):
        """保存专家数据"""
        try:
            with open(self.experts_file, 'w', encoding='utf-8') as f:
                experts_data = {expert_id: expert.to_dict() for expert_id, expert in self.experts.items()}
                json.dump(experts_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存专家数据失败: {e}")
    
    def _save_qa_items(self):
        """保存QA数据"""
        try:
            with open(self.qa_items_file, 'w', encoding='utf-8') as f:
                qa_data = {qa_id: qa_item.to_dict() for qa_id, qa_item in self.qa_items.items()}
                json.dump(qa_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存QA数据失败: {e}")
    
    def _save_tasks(self):
        """保存任务数据"""
        try:
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                tasks_data = {task_id: task.to_dict() for task_id, task in self.annotation_tasks.items()}
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存任务数据失败: {e}")