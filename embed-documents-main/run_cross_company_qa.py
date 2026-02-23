"""Run cross-company questions and save answers."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from rag.retrieval.pipeline import RetrievalPipeline
from rag.retrieval.answer_synthesizer import AnswerSynthesizer
from utils.data_helpers import initialize_metadata_data, initialize_stock_data

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent


async def run_questions():
    """Run all questions from questions_cross_company.json and save answers."""
    
    # Initialize data
    await initialize_stock_data()
    await initialize_metadata_data()
    
    # Load questions
    questions_file = SCRIPT_DIR / "questions_cross_company.json"
    with open(questions_file) as f:
        data = json.load(f)
    
    questions = data["questions"]
    print(f"Loaded {len(questions)} questions")
    
    # Initialize pipeline and synthesizer
    pipeline = RetrievalPipeline()
    synthesizer = AnswerSynthesizer()
    
    answers = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "questions_file": "questions_cross_company.json",
        "answers": []
    }
    
    for i, q in enumerate(questions, 1):
        question_id = q["id"]
        question_text = q["question"]
        
        print(f"\n[{i}/{len(questions)}] {question_id}: {question_text}")
        
        # Build filters if company-specific
        filters = None
        if q.get("fincode"):
            filters = {"fincode": q["fincode"]}
        
        try:
            # Retrieve documents
            result = await pipeline.retrieve(
                query=question_text,
                k=10,
                filters=filters,
            )
            
            if not result.route.use_rag or not result.documents:
                answer_entry = {
                    "id": question_id,
                    "question": question_text,
                    "answer": "No relevant documents found.",
                    "sources": [],
                    "num_chunks_retrieved": 0,
                }
            else:
                # Synthesize answer
                synth_result = await synthesizer.synthesize(
                    query=question_text,
                    documents=result.documents,
                    max_docs=8,
                )
                
                # Extract sources
                sources = []
                for doc in result.documents:
                    source = doc.metadata.get("source") if doc.metadata else None
                    if source and source not in sources:
                        sources.append(source)
                
                answer_entry = {
                    "id": question_id,
                    "question": question_text,
                    "answer": synth_result.answer,
                    "citations": synth_result.citations,
                    "sources": sources,
                    "num_chunks_retrieved": len(result.documents),
                }
                
                print(f"   Answer: {synth_result.answer[:200]}..." if len(synth_result.answer) > 200 else f"   Answer: {synth_result.answer}")
                print(f"   Sources: {len(sources)} documents")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            answer_entry = {
                "id": question_id,
                "question": question_text,
                "answer": None,
                "error": str(e),
            }
        
        answers["answers"].append(answer_entry)
    
    # Save answers
    answers_file = SCRIPT_DIR / "answers_cross_company.json"
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=2)
    
    print(f"\n\nSaved {len(answers['answers'])} answers to {answers_file}")


if __name__ == "__main__":
    asyncio.run(run_questions())
