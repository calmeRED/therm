#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path

def restore_project_from_md(md_file="project_code.md", output_dir="restored_project"):
    md_path = Path(md_file)
    if not md_path.exists():
        print(f"❌ Error: {md_file} not found!")
        return

    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则匹配：标题 + 代码块
    # 匹配形如：## `path/to/file.py` ... ```lang ... ```
    pattern = r'^(#{2,6})\s*`([^`\n]+)`\s*\n\n```(\w*)\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

    if not matches:
        print("⚠️ No code blocks with file paths found in the Markdown file.")
        return

    restored_count = 0
    for _, rel_path_str, lang, code in matches:
        rel_path = Path(rel_path_str)
        target_file = output_root / rel_path

        # 确保目标目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # 写入代码内容（去除末尾可能多余的换行）
        code = code.rstrip('\n')
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"✅ Restored: {rel_path}")
            restored_count += 1
        except OSError as e:
            print(f"❌ Failed to write {rel_path}: {e}")

    print(f"\n�� Done! Restored {restored_count} files to '{output_dir}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Restore project files from a Markdown export.")
    parser.add_argument("-i", "--input", default="project_code.md", help="Input Markdown file (default: project_code.md)")
    parser.add_argument("-o", "--output", default="restored_project", help="Output directory (default: restored_project)")
    args = parser.parse_args()

    restore_project_from_md(md_file=args.input, output_dir=args.output)