"""Generate a text summary of questions and answers from answers_mirae.json."""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    with open(SCRIPT_DIR / 'answers_mirae.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_lines = []
    output_lines.append('=' * 80)
    output_lines.append('SEBI MUTUAL FUNDS REGULATIONS RAG TESTING - QUESTIONS & ANSWERS')
    output_lines.append('=' * 80)
    output_lines.append(f'Generated: {data["generated_at"]}')
    output_lines.append(f'Description: {data["description"]}')
    output_lines.append('')

    for ans in data['answers']:
        output_lines.append('=' * 80)
        output_lines.append(f'ID: {ans["id"]}')
        output_lines.append(f'Category: {ans.get("category", "N/A")}')
        output_lines.append(f'Type: {ans.get("question_type", "N/A")}')
        output_lines.append('')
        output_lines.append(f'QUESTION: {ans["question"]}')
        output_lines.append('')
        output_lines.append('ANSWER:')
        if ans.get('answer'):
            output_lines.append(ans['answer'])
        elif ans.get('error'):
            output_lines.append(f'[ERROR: {ans["error"]}]')
        else:
            output_lines.append('[No answer available]')
        output_lines.append('')
        
        # Citations without bbox
        if ans.get('citations'):
            output_lines.append('CITATIONS:')
            for cit in ans['citations']:
                cit_id = cit.get('id', 'N/A')
                source = cit.get('source', 'N/A')
                page = cit.get('page', 'N/A')
                output_lines.append(f'  [{cit_id}] Source: {source}, Page: {page}')
        
        # Sources
        if ans.get('sources'):
            output_lines.append('')
            output_lines.append(f'SOURCES: {", ".join(ans["sources"])}')
        
        output_lines.append('')

    with open(SCRIPT_DIR / 'qa_mirae_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f'Created qa_mirae_summary.txt with {len(data["answers"])} Q&A entries')


if __name__ == '__main__':
    main()
