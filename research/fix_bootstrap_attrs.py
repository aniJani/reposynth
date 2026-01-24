"""Fix bootstrap CI cell to use correct EvaluationResult attribute names."""
import json

def fix_bootstrap_attrs():
    with open('Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find and fix the bootstrap CI cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source_lines = cell['source']
        modified = False

        for j, line in enumerate(source_lines):
            # Fix wrong attribute names
            if "'answer_similarity'" in line:
                source_lines[j] = line.replace("'answer_similarity'", "'answer_correctness'")
                modified = True
            if "'Answer Similarity'" in line:
                source_lines[j] = source_lines[j].replace("'Answer Similarity'", "'Answer Correctness'")
                modified = True
            if "'keyword_recall'" in line:
                source_lines[j] = source_lines[j].replace("'keyword_recall'", "'context_recall'")
                modified = True
            if "'Keyword Recall'" in line:
                source_lines[j] = source_lines[j].replace("'Keyword Recall'", "'Context Recall'")
                modified = True
            if "'file_recall'" in line:
                source_lines[j] = source_lines[j].replace("'file_recall'", "'context_precision'")
                modified = True
            if "'File Recall'" in line:
                source_lines[j] = source_lines[j].replace("'File Recall'", "'Context Precision'")
                modified = True

        if modified:
            print(f"Fixed cell {i}")
            cell['source'] = source_lines

    with open('Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("\nFixed EvaluationResult attribute names:")
    print("  answer_similarity -> answer_correctness")
    print("  keyword_recall -> context_recall")
    print("  file_recall -> context_precision")

if __name__ == '__main__':
    fix_bootstrap_attrs()
