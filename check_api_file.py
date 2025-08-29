#!/usr/bin/env python3

with open('src/expert_evaluation/api.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    
    print(f'Total lines: {len(lines)}')
    print('Last 10 lines:')
    for i, line in enumerate(lines[-10:], len(lines)-9):
        print(f'{i}: {line}')
    
    # Check if the file ends properly
    if content.endswith('\n'):
        print("File ends with newline")
    else:
        print("File does NOT end with newline")
    
    # Check for any obvious syntax issues
    print(f"File size: {len(content)} characters")
    
    # Try to compile the file
    try:
        compile(content, 'src/expert_evaluation/api.py', 'exec')
        print("File compiles successfully")
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        print(f"Line {e.lineno}: {lines[e.lineno-1] if e.lineno <= len(lines) else 'N/A'}")