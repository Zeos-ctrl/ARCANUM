#!/bin/sh
echo "Super Simple Documentation Generator"
echo "===================================="
echo ""

# Create output directories
mkdir -p docs/api/uml

# Step 1: Generate UML diagrams
echo "Step 1: Generating UML diagrams..."
pyreverse -o png -p project -d docs/api/uml src/*.py 2>/dev/null && echo "  ✓ UML generated" || echo "  ✗ UML failed (install pylint)"

# Step 2: Generate documentation for each Python file
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
            python3 << EOF >> "$output_file"
import sys
import inspect
import importlib.util

# Add src to path
sys.path.insert(0, 'src')

try:
    # Import the module
    spec = importlib.util.spec_from_file_location("$module", "$py_file")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    # Document it
    if mod.__doc__:
        print(mod.__doc__)
        print()
    
    # List classes
    classes = [name for name, obj in inspect.getmembers(mod) if inspect.isclass(obj) and obj.__module__ == "$module"]
    if classes:
        print("## Classes")
        for cls_name in classes:
            print(f"- \`{cls_name}\`")
        print()
    
    # List functions
    functions = [name for name, obj in inspect.getmembers(mod) if inspect.isfunction(obj) and obj.__module__ == "$module"]
    if functions:
        print("## Functions")
        for func_name in functions:
            print(f"- \`{func_name}\`")
        print()
except Exception as e:
    print(f"Could not introspect module: {e}")
EOF
        fi
        
        echo "  ✓ Generated: $output_file"
    fi
done

# Step 3: Create index
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

echo "  ✓ Generated: $index_file"

echo ""
echo "===================================="
echo "Documentation generated successfully!"
echo ""
echo "Files created:"
ls -la docs/api/*.md
echo ""
echo "To view: cat docs/api/index.md"
