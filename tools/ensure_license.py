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

import os

# Read the license header from the file
with open("tools/misc/apache_2_0.txt", "r", encoding="utf-8") as license_file:
    license_lines = [line.strip() for line in license_file.readlines()]
    # Remove empty lines
    license_lines = [line for line in license_lines if line]

# List of file extensions to check
file_extensions = [".py", ".sh", ".ipynb", ".slurm", ".h", ".hpp", ".cu", ".cpp", ".txt"]

file_to_ignore = ["NOTICE.txt"]

def check_license_in_file(file_path):
    """Check if the file contains all lines of the license header"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        year_line = license_lines[0]
        for word in year_line.split():
            # It's okay if the year is not 2025, as long as it has the other words.
            if word in content and word != "2025":
                break
        else:
            return False

        for license_line in license_lines[1:]:
            if license_line not in content:
                return False
        return True


def check_license_in_directory(directory):
    """Check for missing license headers in all files in the directory"""
    files_without_license = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                if os.path.relpath(os.path.join(root, file), directory) in file_to_ignore:
                    continue
                file_path = os.path.join(root, file)
                if not check_license_in_file(file_path):
                    files_without_license.append(file_path)

    return files_without_license


if __name__ == "__main__":
    # Change this to the directory you want to check
    directory_to_check = "."

    missing_license_files = check_license_in_directory(directory_to_check)

    if missing_license_files:
        raise FileNotFoundError("Copyright is missing in the following files:\n" + "\n".join(missing_license_files))
    print("All files have the correct license.")