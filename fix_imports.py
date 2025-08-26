#!/usr/bin/env python3
"""
修复src目录中的相对导入问题
将相对导入改为绝对导入，以便在测试脚本中正常使用
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """修复单个文件中的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复相对导入模式
        patterns = [
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import', r'from \1 import'),
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*)', r'from \1'),
            (r'import \.([a-zA-Z_][a-zA-Z0-9_]*)', r'import \1'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 修复了 {file_path}")
            return True
        else:
            print(f"⚪ {file_path} 无需修复")
            return False
            
    except Exception as e:
        print(f"❌ 修复 {file_path} 失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 修复src目录中的相对导入问题")
    print("=" * 40)
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src目录不存在")
        return False
    
    python_files = list(src_dir.glob("*.py"))
    print(f"📁 找到 {len(python_files)} 个Python文件")
    
    fixed_count = 0
    for file_path in python_files:
        if file_path.name == "__init__.py":
            continue
        
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\n📊 总计修复了 {fixed_count} 个文件")
    
    if fixed_count > 0:
        print("🎉 导入修复完成！现在可以运行测试脚本了。")
    else:
        print("ℹ️ 没有需要修复的文件。")
    
    return True

if __name__ == "__main__":
    main()