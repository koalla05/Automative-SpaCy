#!/usr/bin/env python3
"""
Script to clean canon_models.txt file once.
Converts left side (original names) to cleaned format: only alphanumeric, lowercase.

Usage:
    python clean_canon_models.py

This will:
1. Read canon_models.txt
2. Clean left side of each mapping (remove special chars, lowercase)
3. Create backup: canon_models.txt.backup
4. Overwrite canon_models.txt with cleaned version
"""

import os
import re
import shutil
from datetime import datetime


def clean_model_name(model_name: str) -> str:
    """
    Clean model name: keep only letters and numbers, convert to lowercase.

    Examples:
        "LXP-LB-EU 10k" -> "lxplbeu10k"
        "12/3000/120" -> "123000120"
        "US5000" -> "us5000"
    """
    if not model_name:
        return ""

    # Remove all non-alphanumeric characters, convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', model_name)
    return cleaned.lower()


def clean_canon_file(input_file: str, create_backup: bool = True):
    """
    Clean the canon_models.txt file in-place.

    Args:
        input_file: Path to canon_models.txt
        create_backup: Whether to create backup file
    """
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return

    # Create backup
    if create_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{input_file}.backup_{timestamp}"
        shutil.copy2(input_file, backup_file)
        print(f"✅ Backup created: {backup_file}")

    # Read and process file
    cleaned_lines = []
    skipped_count = 0
    processed_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            raw = line.strip()

            # Keep comments and empty lines as-is
            if not raw or raw.startswith("#"):
                cleaned_lines.append(line)
                continue

            # Process mapping lines
            if "->" not in raw:
                print(f"⚠️  Line {line_num}: Skipping invalid format: {raw}")
                cleaned_lines.append(line)
                skipped_count += 1
                continue

            left, right = raw.split("->", 1)
            original = left.strip()
            canonical = right.strip()

            # Clean the left side
            cleaned_left = clean_model_name(original)

            if not cleaned_left:
                print(f"⚠️  Line {line_num}: Cleaning produced empty string for '{original}', keeping original")
                cleaned_lines.append(line)
                skipped_count += 1
                continue

            # Create new line with cleaned left side
            new_line = f"{cleaned_left} -> {canonical}\n"
            cleaned_lines.append(new_line)
            processed_count += 1

            # Show what changed
            if cleaned_left != original.lower():
                print(f"   Line {line_num}: '{original}' → '{cleaned_left}'")

    # Write cleaned file
    with open(input_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"\n✅ File cleaned successfully!")
    print(f"   Processed: {processed_count} mappings")
    print(f"   Skipped: {skipped_count} lines")
    print(f"   Output: {input_file}")


if __name__ == "__main__":
    # Find canon_models.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try multiple possible locations
    possible_paths = [
        os.path.join(script_dir, "..", "..", "data", "canon_models.txt"),
        os.path.join(script_dir, "data", "canon_models.txt"),
        os.path.join(script_dir, "canon_models.txt"),
    ]

    canon_file = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            canon_file = abs_path
            break

    if not canon_file:
        print("❌ Error: canon_models.txt not found!")
        print("\nSearched in:")
        for path in possible_paths:
            print(f"   - {os.path.abspath(path)}")
        print("\nPlease specify the correct path or run from the correct directory.")
        exit(1)

    print("=" * 70)
    print("CANONICAL MODELS CLEANER")
    print("=" * 70)
    print(f"\nFile: {canon_file}\n")

    # Ask for confirmation
    response = input("This will modify canon_models.txt (backup will be created). Continue? [y/N]: ")

    if response.lower() in ['y', 'yes']:
        clean_canon_file(canon_file, create_backup=True)
    else:
        print("❌ Operation cancelled.")