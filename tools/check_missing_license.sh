#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to check for missing license headers in source files
# Looks for the SPDX-FileCopyrightText in Python and other source files

# File patterns to include (add or remove as needed)
INCLUDE_PATTERNS=(
  "*.py"
  "*.sh"
  "*.ipynb"
  "*.slurm"
  "*.h"
  "*.hpp"
  "*.cu"
  "*.cpp"
)

# Directories to explicitly exclude
EXCLUDE_DIRS=(
  ".git"
)

# Build the find command to get all files but exclude specified directories
FIND_CMD="find . -type f"

# Add include patterns to find
INCLUDE_FIND=""
for pattern in "${INCLUDE_PATTERNS[@]}"; do
  # Convert shell glob to find -name pattern and combine with OR
  if [ -z "$INCLUDE_FIND" ]; then
    INCLUDE_FIND="\\( -name \"$pattern\""
  else
    INCLUDE_FIND="$INCLUDE_FIND -o -name \"$pattern\""
  fi
done
INCLUDE_FIND="$INCLUDE_FIND \\)"
FIND_CMD="$FIND_CMD $INCLUDE_FIND"

# Add exclude directories to find
for dir in "${EXCLUDE_DIRS[@]}"; do
  FIND_CMD="$FIND_CMD -not -path \"./$dir/*\" -not -path \"./$dir\""
done

# Execute the find command to get eligible files
FILES=$(eval "$FIND_CMD")

# First identify all files missing license headers
ALL_MISSING_FILES=()
for file in $FILES; do
  if ! grep -q "SPDX-License-Identifier:" "$file"; then
    ALL_MISSING_FILES+=("$file")
  fi
done

GIT_EXCLUDE_PATTERNS=()

# Read the .gitignore file line by line and skip those comment starting with # and empty lines
if [ -f ".gitignore" ]; then
  while IFS= read -r line; do
    if [[ "$line" == \#* || -z "$line" ]]; then
      continue
    fi
    GIT_EXCLUDE_PATTERNS+=("$line")
  done < ".gitignore"
fi

# Filter out files that match any of the git exclude patterns
FINAL_MISSING_FILES=()
for file in "${ALL_MISSING_FILES[@]}"; do
  SKIP=false
  for pattern in "${GIT_EXCLUDE_PATTERNS[@]}"; do
    # Remove leading and trailing whitespace
    pattern=$(echo "$pattern" | xargs)

    # Check for different gitignore pattern types
    if [[ "$pattern" == *"**/"* ]]; then
      # Handle **/ pattern (matches in any directory)
      base_pattern=${pattern#**/}
      if [[ "$base_pattern" == *"/*" ]]; then
        # Pattern like **/output/* - match directories
        dir_pattern=${base_pattern%/*}
        if [[ "$file" == *"/$dir_pattern/"* ]]; then
          SKIP=true
          echo "Skipping $file (matches pattern $pattern)"
          break
        fi
      elif [[ "$base_pattern" != *"/"* ]]; then
        # Pattern like **/filename - match in any directory
        if [[ "$file" == *"/$base_pattern" ]]; then
          SKIP=true
          echo "Skipping $file (matches pattern $pattern)"
          break
        fi
      else
        # Pattern like **/dir/subdir - exact directory match
        if [[ "$file" == *"/$base_pattern"* ]]; then
          SKIP=true
          echo "Skipping $file (matches pattern $pattern)"
          break
        fi
      fi
    elif [[ "$pattern" == *"/"* ]]; then
      # Patterns with directories
      if [[ "$file" == *"/$pattern"* || "$file" == "./$pattern"* ]]; then
        SKIP=true
        echo "Skipping $file (matches pattern $pattern)"
        break
      fi
    else
      # Simple file patterns (like *.txt)
      if [[ "$file" == *"$pattern" ]]; then
        SKIP=true
        echo "Skipping $file (matches pattern $pattern)"
        break
      fi
    fi
  done

  if [ "$SKIP" = false ]; then
    FINAL_MISSING_FILES+=("$file")
  fi
done

# Count and display final results
COUNT=${#FINAL_MISSING_FILES[@]}
echo "Total files missing license headers: $COUNT"

# Display the list of missing files if there are any
if [ "$COUNT" -gt 0 ]; then
  echo "Files missing license headers:"
  for file in "${FINAL_MISSING_FILES[@]}"; do
    echo "- $file"
  done
fi

# exit with 1 if there are missing license headers
if [ "$COUNT" -gt 0 ]; then
  exit 1
fi
