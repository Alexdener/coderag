#!/usr/bin/env python3
"""
Build Index - Index multi-language code files for semantic search
This script processes Go, Python, Dockerfile, Makefile, and shell script files
and creates a vector index using LlamaIndex and Weaviate.
"""
import logging
import sys
import os
import torch
import weaviate
from pathlib import Path
from typing import List
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Configuration ---
CODE_DIR = os.getenv("CODE_DIR", "your_code_directory")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "CodeIndex")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def split_by_function_or_class(file_path: str) -> List[TextNode]:
    """
    Split file content based on language-specific structures:
    - Python: by functions/classes (def, class)
    - Go: by functions (func)
    - Shell/Dockerfile/Makefile: by functions/sections
    Each TextNode contains the text and metadata (file_path, chunk_index).
    """
    logger.debug(f"Splitting file by language structures: {file_path}")
    nodes = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Determine file type based on extension
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name.lower()
        
        if file_ext == '.py':  # Python files
            nodes = split_python_file(content, file_path)
        elif file_ext == '.go':  # Go files
            nodes = split_go_file(content, file_path)
        elif file_ext in ['.sh', '.bash', '.zsh', '.ksh', '.csh', '.fish']:  # Shell scripts
            nodes = split_shell_file(content, file_path)
        elif file_name.startswith('dockerfile') or file_ext == '.dockerfile':  # Dockerfiles
            nodes = split_dockerfile(content, file_path)
        elif file_ext == '.mk' or file_name in ['makefile', 'makefile.am', 'makefile.in'] or file_name.endswith('makefile'):  # Makefiles
            nodes = split_makefile(content, file_path)
        else:  # Default: split by significant comment blocks or treat as single chunk
            nodes = split_generic_file(content, file_path)

    except Exception as e:
        logger.error(f"Error reading or splitting file {file_path}: {e}")
        # Return single node with entire content if processing fails
        nodes = [TextNode(
            text=content.strip() if 'content' in locals() else "",
            metadata={"file_path": str(file_path), "chunk_index": 0, "type": "error_processed"}
        )]

    logger.info(f"Split {file_path} into {len(nodes)} TextNodes using language-specific logic.")
    return nodes


def split_python_file(content: str, file_path: str) -> List[TextNode]:
    """Split Python file by functions and classes."""
    import re
    
    # Pattern to match Python functions and classes
    # This pattern matches def/class lines and their content until the next def/class or end
    pattern = r'^(def|class)\s+.*?:\s*\n(?:\s+.*?\n?)*?(?=\n^(def|class)\s+|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    nodes = []
    last_end = 0
    
    # Find all function/class definitions
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        start, end = match.span()
        
        # Add content before this function/class if it exists and is significant
        if start > last_end:
            prefix_content = content[last_end:start].strip()
            if prefix_content:
                nodes.append(TextNode(
                    text=prefix_content,
                    metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "python_prefix"}
                ))
        
        # Add the function/class content
        func_class_content = content[start:end].strip()
        if func_class_content:
            nodes.append(TextNode(
                text=func_class_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "python_function_or_class"}
            ))
        
        last_end = end
    
    # Add any remaining content after the last function/class
    if last_end < len(content):
        suffix_content = content[last_end:].strip()
        if suffix_content:
            nodes.append(TextNode(
                text=suffix_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "python_suffix"}
            ))
    
    # If no functions or classes found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "python_full_file"}
        )]
    
    return nodes


def split_go_file(content: str, file_path: str) -> List[TextNode]:
    """Split Go file by functions."""
    import re
    
    # Pattern to match Go functions
    # This pattern matches func lines and their content until the closing brace
    pattern = r'^(func)\s+.*?{(?:[^{}]|\{[^{}]*\})*}(?=\n^func\s+|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    nodes = []
    last_end = 0
    
    # Find all function definitions
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        start, end = match.span()
        
        # Add content before this function if it exists and is significant
        if start > last_end:
            prefix_content = content[last_end:start].strip()
            if prefix_content:
                nodes.append(TextNode(
                    text=prefix_content,
                    metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "go_prefix"}
                ))
        
        # Add the function content
        func_content = content[start:end].strip()
        if func_content:
            nodes.append(TextNode(
                text=func_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "go_function"}
            ))
        
        last_end = end
    
    # Add any remaining content after the last function
    if last_end < len(content):
        suffix_content = content[last_end:].strip()
        if suffix_content:
            nodes.append(TextNode(
                text=suffix_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "go_suffix"}
            ))
    
    # If no functions found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "go_full_file"}
        )]
    
    return nodes


def split_shell_file(content: str, file_path: str) -> List[TextNode]:
    """Split shell script by functions."""
    import re
    
    # Pattern to match shell functions (function name() or name() format)
    pattern = r'^(\w+)\s*\(\)\s*\{(?:[^{}]|\{[^{}]*\})*\}(?=\n^\w+\s*\(\)\s*\{|$)'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    nodes = []
    last_end = 0
    
    # Find all function definitions
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        start, end = match.span()
        
        # Add content before this function if it exists and is significant
        if start > last_end:
            prefix_content = content[last_end:start].strip()
            if prefix_content:
                nodes.append(TextNode(
                    text=prefix_content,
                    metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "shell_prefix"}
                ))
        
        # Add the function content
        func_content = content[start:end].strip()
        if func_content:
            nodes.append(TextNode(
                text=func_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "shell_function"}
            ))
        
        last_end = end
    
    # Add any remaining content after the last function
    if last_end < len(content):
        suffix_content = content[last_end:].strip()
        if suffix_content:
            nodes.append(TextNode(
                text=suffix_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "shell_suffix"}
            ))
    
    # If no functions found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "shell_full_file"}
        )]
    
    return nodes


def split_dockerfile(content: str, file_path: str) -> List[TextNode]:
    """Split Dockerfile by instruction blocks."""
    import re
    
    # Pattern to match Dockerfile instructions
    lines = content.split('\n')
    nodes = []
    current_block = []
    current_instruction = ""
    
    for line in lines:
        # Check if this line starts a new instruction
        instruction_match = re.match(r'^\s*(\w+)\s+(.*)', line, re.IGNORECASE)
        if instruction_match:
            instruction = instruction_match.group(1).upper()
            
            # If we have accumulated content for a different instruction, save it
            if current_block and instruction != current_instruction:
                if current_instruction:  # Don't save if this is the first instruction
                    block_content = '\n'.join(current_block).strip()
                    if block_content:
                        nodes.append(TextNode(
                            text=block_content,
                            metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "dockerfile_block"}
                        ))
                
                # Start new block
                current_block = [line]
                current_instruction = instruction
            else:
                current_block.append(line)
        else:
            # Continuation of current instruction or comment
            current_block.append(line)
    
    # Add the last block
    if current_block:
        block_content = '\n'.join(current_block).strip()
        if block_content:
            nodes.append(TextNode(
                text=block_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "dockerfile_block"}
            ))
    
    # If no blocks found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "dockerfile_full_file"}
        )]
    
    return nodes


def split_makefile(content: str, file_path: str) -> List[TextNode]:
    """Split Makefile by targets."""
    import re
    
    # Pattern to match Makefile targets (lines ending with : but not in recipes)
    lines = content.split('\n')
    nodes = []
    current_target = []
    current_target_name = ""
    in_recipe = False  # Track if we're inside a recipe (indented commands)
    
    for line in lines:
        stripped_line = line.strip()
        
        # Check if this line is a target (ends with : but not in recipe)
        if not in_recipe and stripped_line.endswith(':') and not line.startswith('\t') and not line.startswith(' '):
            # This is a new target
            if current_target and current_target_name:
                # Save the previous target
                target_content = '\n'.join(current_target).strip()
                if target_content:
                    nodes.append(TextNode(
                        text=target_content,
                        metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "makefile_target"}
                    ))
            
            # Start new target
            current_target = [line]
            current_target_name = stripped_line
            in_recipe = False
        else:
            # Part of current target (recipe or dependency)
            current_target.append(line)
            # Check if this line starts a recipe (tab-indented command)
            if line.startswith('\t') or (line.startswith(' ') and line.lstrip().startswith('\t')):
                in_recipe = True
            elif stripped_line == '' or stripped_line.startswith('#'):
                # Empty line or comment doesn't affect recipe state
                pass
            elif not stripped_line.endswith(':'):
                # Non-indented non-target line might end recipe
                in_recipe = False
    
    # Add the last target
    if current_target and current_target_name:
        target_content = '\n'.join(current_target).strip()
        if target_content:
            nodes.append(TextNode(
                text=target_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "makefile_target"}
            ))
    
    # If no targets found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "makefile_full_file"}
        )]
    
    return nodes


def split_generic_file(content: str, file_path: str) -> List[TextNode]:
    """Generic splitting for unrecognized file types."""
    # For files without specific logic, split by significant comment blocks
    import re
    
    # Look for comment blocks (various formats)
    comment_patterns = [
        r'(/\*[\s\S]*?\*/)',  # C-style /* */ comments
        r'(#[\s\S]*?)(?=\n\S|\n{2,}|$)',  # Shell/Python-style # comments (until non-comment line or double newline)
        r'(--[\s\S]*?)(?=\n\S|\n{2,}|$)',  # SQL/Haskell-style -- comments
        r'(\{-[\s\S]*?-\})'  # Haskell-style {- -} comments
    ]
    
    nodes = []
    last_end = 0
    
    # Try to find comment blocks
    for pattern in comment_patterns:
        for match in re.finditer(pattern, content):
            start, end = match.span()
            
            # Add content before comment if significant
            if start > last_end:
                prefix_content = content[last_end:start].strip()
                if prefix_content:
                    nodes.append(TextNode(
                        text=prefix_content,
                        metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "generic_prefix"}
                    ))
            
            # Add the comment content
            comment_content = content[start:end].strip()
            if comment_content:
                nodes.append(TextNode(
                    text=comment_content,
                    metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "generic_comment"}
                ))
            
            last_end = end
    
    # Add any remaining content
    if last_end < len(content):
        suffix_content = content[last_end:].strip()
        if suffix_content:
            nodes.append(TextNode(
                text=suffix_content,
                metadata={"file_path": file_path, "chunk_index": len(nodes), "type": "generic_suffix"}
            ))
    
    # If no comment blocks found, treat as single chunk
    if not nodes:
        nodes = [TextNode(
            text=content.strip(),
            metadata={"file_path": file_path, "chunk_index": 0, "type": "generic_full_file"}
        )]
    
    return nodes

def main():
    logger.info(f"Starting index build process...")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Code directory: {CODE_DIR}")
    logger.info(f"Weaviate endpoint: http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    logger.info(f"Weaviate class name: {WEAVIATE_CLASS_NAME}")

    # Connect to Weaviate first
    logger.info("Connecting to Weaviate...")
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )

    try:
        # Configure embedding model
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            device=DEVICE,
            cache_folder=CACHE_DIR,
            normalize=True,
            max_length=512
        )

        # Delete existing collection if it exists
        if client.collections.exists(WEAVIATE_CLASS_NAME):
            logger.warning(f"Collection '{WEAVIATE_CLASS_NAME}' already exists and will be replaced!")
            logger.info(f"Existing collection will be deleted and recreated with content from: {CODE_DIR}")
            client.collections.delete(WEAVIATE_CLASS_NAME)
        else:
            logger.info(f"Creating new collection: {WEAVIATE_CLASS_NAME}")

        logger.info(f"Scanning documents in: {CODE_DIR}")

        # Define directories to exclude
        exclude_dirs = {'.git', '.svn', '.hg', '.history', '__pycache__', '.venv', 'node_modules', '.vscode', '.idea'}

        # Find multi-language code files to process, excluding certain directories
        all_code_files = []

        # Walk through the directory tree manually to skip excluded directories
        for root, dirs, files in os.walk(CODE_DIR):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            # Check for files with specific extensions
            for ext in ("*.py", "*.go", "*.sh", "*.bash", "*.zsh", "*.dockerfile", "*.mk"):
                for file_path in Path(root).glob(ext):
                    if file_path.is_file():
                        all_code_files.append(file_path)

        # Also include files named Dockerfile or Makefile regardless of extension
        for root, dirs, files in os.walk(CODE_DIR):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file_name in files:
                if file_name.lower().startswith('dockerfile') or file_name.lower().startswith('makefile'):
                    file_path = Path(root) / file_name
                    if file_path.is_file():
                        # Avoid duplicates if already added by extension
                        if file_path not in all_code_files:
                            all_code_files.append(file_path)

        logger.info(f"Found {len(all_code_files)} code files to process.")
        if not all_code_files:
            logger.warning("No source files found.")
            return  # Return instead of sys.exit to ensure client is closed

        # Process all files and create nodes
        all_nodes = []
        for i, file_path_obj in enumerate(all_code_files):
            file_path_str = str(file_path_obj)
            logger.info(f"Processing file {i+1}/{len(all_code_files)}: {file_path_str}")

            # Use our language-specific splitting function
            file_nodes = split_by_function_or_class(file_path_str)
            all_nodes.extend(file_nodes)

        logger.info(f"Generated a total of {len(all_nodes)} nodes using custom splitting.")
        if not all_nodes:
            logger.warning("No nodes were generated.")
            return  # Return instead of sys.exit to ensure client is closed

        # Build the index
        logger.info("Building vector index...")
        # Use the correct constructor for WeaviateVectorStore
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=WEAVIATE_CLASS_NAME,
            text_key="content"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True
        )

        logger.info("Index building process completed successfully.")
        logger.info(f"Created index with {len(all_nodes)} nodes in collection '{WEAVIATE_CLASS_NAME}'")

    finally:
        # Always close client connection
        client.close()
        logger.info("Weaviate client connection closed.")

if __name__ == "__main__":
    main()