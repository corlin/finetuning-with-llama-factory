"""
专家评估系统单元测试模块

本模块包含专家评估系统所有组件的单元测试，确保系统的可靠性和正确性。
"""

# 测试工具和配置
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 测试配置
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# 确保测试目录存在
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"