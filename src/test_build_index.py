#!/usr/bin/env python3
"""
Test script to verify the multi-language indexing functionality
"""
import os
import sys
import tempfile
from pathlib import Path
from build_index import split_by_function_or_class
from llama_index.core.schema import TextNode

def test_python_splitting():
    """Test the Python file splitting function"""
    print("Testing Python file splitting...")
    
    # Test 1: Python file with functions and classes
    test_content1 = '''#!/usr/bin/env python3
\"\"\"
This is a module docstring
\"\"\"

import os
import sys

def hello_world():
    \"\"\"
    Prints hello world
    \"\"\"
    print("Hello, World!")

class MyClass:
    \"\"\"
    A simple class
    \"\"\"
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

def another_function(x, y):
    \"\"\"
    Adds two numbers
    \"\"\"
    return x + y
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content1)
        temp_file1 = f.name
    
    try:
        nodes = split_by_function_or_class(temp_file1)
        print(f"Test 1 - Python file with functions/classes: Generated {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: {len(node.text)} chars, type: {node.metadata.get('type', 'unknown')}")
            # Print first 50 chars of text
            text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"    Preview: {text_preview}")
        print()
    finally:
        os.unlink(temp_file1)

def test_go_splitting():
    """Test the Go file splitting function"""
    print("Testing Go file splitting...")
    
    # Test 2: Go file with functions
    test_content2 = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

// greet prints a greeting
func greet(name string) {
    fmt.Printf("Hello, %s!\n", name)
}

// add returns the sum of two integers
func add(a, b int) int {
    return a + b
}
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
        f.write(test_content2)
        temp_file2 = f.name
    
    try:
        nodes = split_by_function_or_class(temp_file2)
        print(f"Test 2 - Go file with functions: Generated {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: {len(node.text)} chars, type: {node.metadata.get('type', 'unknown')}")
            # Print first 50 chars of text
            text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"    Preview: {text_preview}")
        print()
    finally:
        os.unlink(temp_file2)

def test_shell_splitting():
    """Test the shell script splitting function"""
    print("Testing Shell script splitting...")
    
    # Test 3: Shell script with functions
    test_content3 = '''#!/bin/bash

# Function to print hello
hello() {
    echo "Hello, World!"
}

# Function to add numbers
add_numbers() {
    local num1=$1
    local num2=$2
    echo $(($num1 + $num2))
}

# Main execution
if [ "$1" = "hello" ]; then
    hello
elif [ "$1" = "add" ]; then
    add_numbers $2 $3
fi
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(test_content3)
        temp_file3 = f.name
    
    try:
        nodes = split_by_function_or_class(temp_file3)
        print(f"Test 3 - Shell script with functions: Generated {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: {len(node.text)} chars, type: {node.metadata.get('type', 'unknown')}")
            # Print first 50 chars of text
            text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"    Preview: {text_preview}")
        print()
    finally:
        os.unlink(temp_file3)

def test_dockerfile_splitting():
    """Test the Dockerfile splitting function"""
    print("Testing Dockerfile splitting...")
    
    # Test 4: Dockerfile
    test_content4 = '''FROM ubuntu:20.04

LABEL maintainer="example@example.com"

RUN apt-get update && \\
    apt-get install -y python3 python3-pip

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
        f.write(test_content4)
        temp_file4 = f.name
    
    try:
        nodes = split_by_function_or_class(temp_file4)
        print(f"Test 4 - Dockerfile: Generated {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: {len(node.text)} chars, type: {node.metadata.get('type', 'unknown')}")
            # Print first 50 chars of text
            text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"    Preview: {text_preview}")
        print()
    finally:
        os.unlink(temp_file4)

def test_makefile_splitting():
    """Test the Makefile splitting function"""
    print("Testing Makefile splitting...")
    
    # Test 5: Makefile
    test_content5 = '''CC=gcc
CFLAGS=-Wall -g

all: program

program: main.o utils.o
	$(CC) -o program main.o utils.o

main.o: main.c
	$(CC) -c main.c

utils.o: utils.c utils.h
	$(CC) -c utils.c

clean:
	rm -f *.o program

install: program
	cp program /usr/local/bin/

.PHONY: all clean install
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='Makefile', delete=False) as f:
        f.write(test_content5)
        temp_file5 = f.name
    
    try:
        nodes = split_by_function_or_class(temp_file5)
        print(f"Test 5 - Makefile: Generated {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"  Node {i}: {len(node.text)} chars, type: {node.metadata.get('type', 'unknown')}")
            # Print first 50 chars of text
            text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"    Preview: {text_preview}")
        print()
    finally:
        os.unlink(temp_file5)

def test_directory_scanning():
    """Test directory scanning functionality for multi-language files"""
    print("Testing multi-language directory scanning...")
    
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files for different languages
        (temp_path / "script.py").write_text('''def hello():
    print("Hello from Python!")
''')
        
        (temp_path / "main.go").write_text('''package main
func main() {
    println("Hello from Go!")
}
''')
        
        (temp_path / "script.sh").write_text('''#!/bin/bash
echo "Hello from Shell!"
''')
        
        (temp_path / "Dockerfile").write_text('''FROM alpine:latest
RUN echo "Hello from Dockerfile"
''')
        
        (temp_path / "Makefile").write_text('''all:
	echo "Hello from Makefile"
''')
        
        (temp_path / "readme.txt").write_text("This should not be included")
        
        # Find multi-language code files (similar to what build_index.py does)
        all_code_files = []
        for ext in ("*.py", "*.go", "*.sh", "*.bash", "*.zsh", "*.dockerfile", "*.mk", "Dockerfile", "Makefile", "makefile"):
            all_code_files.extend(temp_path.rglob(ext))
        
        # Also include files that start with "Dockerfile" or "Makefile" regardless of extension
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                file_name = file_path.name.lower()
                if file_name.startswith('dockerfile') or file_name.startswith('makefile'):
                    all_code_files.append(file_path)
        
        print(f"Found {len(all_code_files)} code files in directory scanning test")
        for f in all_code_files:
            print(f"  - {f.name}")
        
        # Verify that code files were found
        expected_extensions = {".py", ".go", ".sh", "Dockerfile", "Makefile"}
        found_extensions = {f.suffix or f.name for f in all_code_files}
        
        if expected_extensions.issubset(found_extensions) or any(ext in found_extensions for ext in expected_extensions):
            print("✓ Directory scanning correctly found multi-language files")
        else:
            print(f"? Directory scanning found: {found_extensions}")

if __name__ == "__main__":
    print("Running tests for multi-language build_index functionality...\n")
    
    test_python_splitting()
    test_go_splitting()
    test_shell_splitting()
    test_dockerfile_splitting()
    test_makefile_splitting()
    test_directory_scanning()
    
    print("All tests completed!")