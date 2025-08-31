#!/bin/sh
echo "Super Simple Documentation Generator"
echo "===================================="
echo ""

# Create output directories
mkdir -p docs/api/uml

# Generate UML diagrams
echo "Step 1: Generating UML diagrams..."
pyreverse -o png -p project -d docs/api/uml src/*.py 2>/dev/null && echo "UML generated" || echo "UML failed (install pylint)"

# Generate documentation for each Python file
echo ""
echo "Step 2: Generating documentation..."
echo ""

for py_file in src/*.py; do
    if [ -f "$py_file" ]; then
        module=$(basename "$py_file" .py)
        output_file="docs/api/${module}.md"
        
        echo "Processing $module..."
        
        # Create basic markdown header
        echo "# Module: $module" > "$output_file"
        echo "" >> "$output_file"
        echo "Source: \`$py_file\`" >> "$output_file"
        echo "" >> "$output_file"
        
        # Add UML if it exists
        if [ -f "docs/api/uml/classes_project.png" ]; then
            echo "![UML Diagram](uml/classes_project.png)" >> "$output_file"
            echo "" >> "$output_file"
        fi
        
        # Try pydoc-markdown (might work, might not)
        echo "## Documentation" >> "$output_file"
        echo "" >> "$output_file"
        
        # Method 1: Try pydoc-markdown
        if command -v pydoc-markdown &> /dev/null; then
            pydoc-markdown -m "$module" -I src >> "$output_file" 2>/dev/null
        fi
        
        # Method 2: If file is still small, use Python introspection
        if [ $(wc -l < "$output_file") -lt 10 ]; then
            echo "Using Python introspection..."
            MODULE_NAME="$module" PY_FILE_PATH="$py_file" python3 << 'PYTHON_EOF' >> "$output_file"
import ast
import inspect
import os

# Get variables from environment
module_name = os.environ.get('MODULE_NAME', '')
py_file_path = os.environ.get('PY_FILE_PATH', '')

try:
    # Read and parse the Python file using AST
    with open(py_file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the AST
    tree = ast.parse(source_code)
    
    # Extract module docstring
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        print(module_docstring)
        print()
    
    # Find classes and functions
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level functions only
            functions.append(node)
    
    # Process classes
    if classes:
        print('## Classes')
        print()
        for cls_node in classes:
            print('## ' + cls_node.name)
            print()
            
            # Class description
            cls_docstring = ast.get_docstring(cls_node)
            if cls_docstring:
                print('**Description**: ' + cls_docstring)
                print()
            
            # Find constructor
            init_method = None
            methods = []
            for item in cls_node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == '__init__':
                        init_method = item
                    elif not item.name.startswith('_'):
                        methods.append(item)
            
            # Constructor details
            if init_method:
                print('### Constructor')
                print()
                print('```python')
                
                # Build constructor signature
                args = []
                for arg in init_method.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        if hasattr(arg.annotation, 'id'):
                            arg_str += ': ' + arg.annotation.id
                        elif hasattr(arg.annotation, 'attr'):
                            # Handle things like Optional[nn.Module]
                            arg_str += ': ' + ast.unparse(arg.annotation)
                    args.append(arg_str)
                
                # Add defaults
                defaults = init_method.args.defaults
                if defaults:
                    num_defaults = len(defaults)
                    for i, default in enumerate(defaults):
                        arg_index = len(args) - num_defaults + i
                        if hasattr(default, 'value'):
                            args[arg_index] += ' = ' + str(default.value)
                        elif hasattr(default, 'id'):
                            args[arg_index] += ' = ' + default.id
                        else:
                            args[arg_index] += ' = ' + ast.unparse(default)
                
                # Return annotation
                return_annotation = ''
                if init_method.returns:
                    if hasattr(init_method.returns, 'id'):
                        return_annotation = ' -> ' + init_method.returns.id
                    else:
                        return_annotation = ' -> ' + ast.unparse(init_method.returns)
                
                print('def __init__(' + ', '.join(args) + ')' + return_annotation + ':')
                print('```')
                print()
                
                # Constructor docstring
                init_docstring = ast.get_docstring(init_method)
                if init_docstring:
                    print(init_docstring)
                    print()
            
            # Methods table
            if methods:
                print('### Methods')
                print()
                print('| Signature | Description |')
                print('|-----------|-------------|')
                
                for method in methods:
                    # Build method signature
                    args = []
                    for arg in method.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            if hasattr(arg.annotation, 'id'):
                                arg_str += ': ' + arg.annotation.id
                            else:
                                arg_str += ': ' + ast.unparse(arg.annotation)
                        args.append(arg_str)
                    
                    # Add defaults
                    defaults = method.args.defaults
                    if defaults:
                        num_defaults = len(defaults)
                        for i, default in enumerate(defaults):
                            arg_index = len(args) - num_defaults + i
                            if hasattr(default, 'value'):
                                args[arg_index] += ' = ' + str(default.value)
                            elif hasattr(default, 'id'):
                                args[arg_index] += ' = ' + default.id
                            else:
                                args[arg_index] += ' = ' + ast.unparse(default)
                    
                    # Return annotation
                    return_annotation = ''
                    if method.returns:
                        if hasattr(method.returns, 'id'):
                            return_annotation = ' -> ' + method.returns.id
                        else:
                            return_annotation = ' -> ' + ast.unparse(method.returns)
                    
                    signature = method.name + '(' + ', '.join(args) + ')' + return_annotation
                    
                    # Get method description
                    description = 'No description available.'
                    method_docstring = ast.get_docstring(method)
                    if method_docstring:
                        # Get first line of docstring as description
                        description = method_docstring.split('\n')[0].strip()
                    
                    print('| `' + signature + '` | ' + description + ' |')
                print()
    
    # Process functions
    if functions:
        print('## Functions')
        print()
        for func_node in functions:
            print('## ' + func_node.name)
            print()
            
            print('```python')
            
            # Build function signature
            args = []
            for arg in func_node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    if hasattr(arg.annotation, 'id'):
                        arg_str += ': ' + arg.annotation.id
                    else:
                        arg_str += ': ' + ast.unparse(arg.annotation)
                args.append(arg_str)
            
            # Add defaults
            defaults = func_node.args.defaults
            if defaults:
                num_defaults = len(defaults)
                for i, default in enumerate(defaults):
                    arg_index = len(args) - num_defaults + i
                    if hasattr(default, 'value'):
                        args[arg_index] += ' = ' + str(default.value)
                    elif hasattr(default, 'id'):
                        args[arg_index] += ' = ' + default.id
                    else:
                        args[arg_index] += ' = ' + ast.unparse(default)
            
            # Return annotation
            return_annotation = ''
            if func_node.returns:
                if hasattr(func_node.returns, 'id'):
                    return_annotation = ' -> ' + func_node.returns.id
                else:
                    return_annotation = ' -> ' + ast.unparse(func_node.returns)
            
            print('def ' + func_node.name + '(' + ', '.join(args) + ')' + return_annotation + ':')
            print('```')
            print()
            
            # Get docstring
            func_docstring = ast.get_docstring(func_node)
            if func_docstring:
                print('**Description**: ' + func_docstring)
                print()
            else:
                print('**Description**: No description available.')
                print()
            
            print('---')
            print()

except Exception as e:
    print('Could not parse module: ' + str(e))
    print()
    print('**Note:** This module may have syntax errors or use unsupported Python features.')
PYTHON_EOF
        fi
        
        echo "Generated: $output_file"
    fi
done

# Create index
echo ""
echo "Step 3: Creating index..."
index_file="docs/api/index.md"

echo "# Documentation Index" > "$index_file"
echo "" >> "$index_file"
echo "Generated: $(date)" >> "$index_file"
echo "" >> "$index_file"
echo "## Modules" >> "$index_file"
echo "" >> "$index_file"

for md_file in docs/api/*.md; do
    if [ "$md_file" != "$index_file" ] && [ -f "$md_file" ]; then
        name=$(basename "$md_file" .md)
        echo "- [$name]($name.md)" >> "$index_file"
    fi
done

echo "" >> "$index_file"
echo "## UML Diagrams" >> "$index_file"
echo "" >> "$index_file"
for png_file in docs/api/uml/*.png; do
    if [ -f "$png_file" ]; then
        name=$(basename "$png_file")
        echo "- [$name](uml/$name)" >> "$index_file"
    fi
done

echo "Generated: $index_file"

echo ""
echo "===================================="
echo "Documentation generated successfully!"
echo ""
echo "Files created:"
ls -la docs/api/*.md
echo ""
echo "To view: cat docs/api/index.md"
