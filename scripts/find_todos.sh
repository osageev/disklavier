#!/bin/bash
# This script iterates over all Python files in a provided git repository,
# collects all TODO comments (lines starting with "# TODO"), formats them,
# and appends the list to the bottom of the repository's README.md file.
#
# It skips `*.venv` and `*.vscode` directories.
#
# Usage:
#   ./find_todos.sh /path/to/git/repository

# Exit immediately if any command fails.
set -e

# Check if a repository path was provided.
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/git/repository"
    exit 1
fi

REPO_PATH="$1"

# Verify the provided directory exists.
if [ ! -d "$REPO_PATH" ]; then
    echo "Error: Directory '$REPO_PATH' does not exist."
    exit 1
fi

# Check if the directory is a git repository.
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "Error: '$REPO_PATH' is not a git repository."
    exit 1
fi

# Change to the repository directory.
cd "$REPO_PATH" || exit 1

# Ensure README.md exists; if not, create an empty one.
if [ ! -f "README.md" ]; then
    echo "README.md not found. Creating one..."
    touch README.md
fi

# Create a temporary file to store formatted TODOs.
TEMP_TODO_FILE=$(mktemp)

# Add a header for the TODO section.
echo -e "\n\n## TODO List\n" >> "$TEMP_TODO_FILE"

# Search for Python files while excluding ".venv" and ".vscode" directories.
echo "Searching for Python files..."
# FILES=($(find . -type f -name "*.py" -not -path "*/.venv/*" -not -path "*/.vscode/*"))
mapfile -t FILES < <(find . -type f -name "*.py" -not -path "*/.venv/*" -not -path "*/.vscode/*")
TOTAL_FILES=${#FILES[@]}
echo "Found $TOTAL_FILES Python file(s)."

# Process each Python file and extract TODO lines.
INDEX=1
for file in "${FILES[@]}"; do
    echo "Processing file $INDEX of $TOTAL_FILES: $file"
    # Extract lines containing "# TODO", including filename and line number.
    TODO_LINES=$(grep -Hn "# TODO" "$file" 2>/dev/null || true)
    if [ -n "$TODO_LINES" ]; then
        # Format each found TODO line into a bullet list item.
        while IFS= read -r line; do
            echo "- ${line}" >> "$TEMP_TODO_FILE"
        done <<< "$TODO_LINES"
    fi
    INDEX=$((INDEX + 1))
done

# Check if any TODOs were found (the file will have only header if none were found).
if [ "$(wc -l < "$TEMP_TODO_FILE")" -le 2 ]; then
    echo "No TODOs found in Python files."
    rm "$TEMP_TODO_FILE"
    exit 0
fi

# Append the formatted TODO list to the bottom of README.md.
echo "Appending TODOs to README.md..."
cat "$TEMP_TODO_FILE" >> README.md

# Clean up the temporary file.
rm "$TEMP_TODO_FILE"

echo "Formatted TODOs have been appended to README.md successfully."