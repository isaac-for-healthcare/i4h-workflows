import argparse

import toml


def extract_dependencies(toml_file, output_file):
    # Load the TOML file
    with open(toml_file, "r") as file:
        data = toml.load(file)

    # Extract dependencies
    dependencies = data["project"]["dependencies"]
    # Filter out "openpi-client"
    filtered_dependencies = [dep for dep in dependencies if "openpi-client" not in dep]

    # Write to requirements.txt
    with open(output_file, "w") as file:
        for dep in filtered_dependencies:
            # Remove any extra characters like quotes
            dep = dep.strip('"').strip("'")
            file.write(f"{dep}\n")


def add_lerobot_dependency(output_file):
    lerobot_dependency = "git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56"

    # Append the LeRobot dependency to the requirements.txt
    with open(output_file, "a") as file:
        file.write(f"{lerobot_dependency}\n")


def main():
    parser = argparse.ArgumentParser(description="Process pyproject.toml and requirements.txt files.")
    parser.add_argument("toml_file", type=str, help="Path to the pyproject.toml file")
    parser.add_argument("output_file", type=str, help="Path to the output requirements.txt file")
    parser.add_argument("--add-lerobot", action="store_true", help="Add LeRobot dependency to requirements.txt")

    args = parser.parse_args()

    extract_dependencies(args.toml_file, args.output_file)

    if args.add_lerobot:
        add_lerobot_dependency(args.output_file)


if __name__ == "__main__":
    main()
