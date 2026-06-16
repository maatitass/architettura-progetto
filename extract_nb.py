import json

def process_notebook():
    with open('/Users/matiasbonoli/Downloads/architettura/ProgettoMLbonni_2.ipynb', 'r') as f:
        nb = json.load(f)
    
    with open('/Users/matiasbonoli/Downloads/architettura/notebook_content.md', 'w') as f:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            source = "".join(cell.get('source', []))
            
            f.write(f"### Cell {i} ({cell_type})\n")
            if cell_type == 'code':
                f.write("```python\n")
            f.write(source)
            if cell_type == 'code':
                f.write("\n```")
            f.write("\n\n")

if __name__ == '__main__':
    process_notebook()
