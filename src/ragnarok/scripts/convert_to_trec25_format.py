#!/usr/bin/env python3
"""
Script to convert JSONL format from old structure to new structure.
"""

import argparse
import json
import sys
from typing import Dict, List, Any, Optional

def load_prompts_from_file(prompt_file: str) -> Dict[str, str]:
    """
    Load prompts from a separate JSONL file and create a mapping from qid to prompt.
    
    Args:
        prompt_file: Path to the prompt JSONL file
        
    Returns:
        Dictionary mapping qid to prompt text
    """
    prompts = {}
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    qid = record.get("query", {}).get("qid")
                    prompt = record.get("rag_exec_summary", {}).get("prompt")
                    
                    if qid and prompt:
                        prompts[str(qid)] = prompt
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing JSON in prompt file on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing prompt record on line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: Prompt file '{prompt_file}' not found. Proceeding without external prompts.")
    except Exception as e:
        print(f"Warning: Error reading prompt file '{prompt_file}': {e}")
    
    print(f"Loaded {len(prompts)} prompts from external file.")
    return prompts

def convert_record(old_record: Dict[str, Any], prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Convert a single record from old format to new format.
    
    Args:
        old_record: Dictionary containing the old format record
        prompts: Optional dictionary mapping qid to prompt text
        
    Returns:
        Dictionary in the new format structure
    """
    
    # Extract metadata fields
    metadata = {
        "team_id": old_record.get("team_id", "organizer"),
        "run_id": old_record.get("run_id", "unknown-run"),
        "type": old_record.get("type", "automatic"),
        "narrative_id": old_record.get("topic_id", old_record.get("narrative_id", 1)),
        "narrative": old_record.get("topic", old_record.get("narrative", ""))
    }
    
    # Add prompt field only if external prompt file is provided and contains the prompt
    topic_id = str(old_record.get("topic_id", old_record.get("narrative_id", "")))
    
    if prompts and topic_id in prompts:
        # Use prompt from external file
        metadata["prompt"] = prompts[topic_id]
    
    # Handle references - ensure it's a list
    references = old_record.get("references", [])
    if not isinstance(references, list):
        references = []
    
    # Convert answer format
    answer = []
    
    if "answer" in old_record:
        old_answer = old_record["answer"]
        answer = old_answer
    
    # If no answer was found, create a placeholder
    if not answer:
        answer.append({
            "text": "No answer content available in the original record.",
            "citations": []
        })
    
    # Construct the new format
    new_record = {
        "metadata": metadata,
        "references": references,
        "answer": old_answer
    }
    
    return new_record

def convert_jsonl_file(input_file: str, output_file: str, prompt_file: Optional[str] = None, verbose: bool = False):
    """
    Convert entire JSONL file from old format to new format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        prompt_file: Optional path to prompt JSONL file
        verbose: Whether to show detailed progress information
    """
    
    # Load prompts if prompt file is provided
    prompts = None
    if prompt_file:
        prompts = load_prompts_from_file(prompt_file)
    
    converted_count = 0
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse the JSON record
                    old_record = json.loads(line)
                    
                    # Convert to new format
                    new_record = convert_record(old_record, prompts)
                    
                    # Write to output file
                    json.dump(new_record, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    converted_count += 1
                    
                    if verbose and converted_count % 100 == 0:
                        print(f"Converted {converted_count} records...")
                
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"Error parsing JSON on line {line_num}: {e}")
                    error_count += 1
                    continue
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing record on line {line_num}: {e}")
                    error_count += 1
                    continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)
    
    print(f"Conversion completed!")
    print(f"Successfully converted: {converted_count} records")
    if error_count > 0:
        print(f"Errors encountered: {error_count} records")
    print(f"Output written to: {output_file}")

def parse_arguments():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Convert JSONL format from old structure to new structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.jsonl converted_data.jsonl
  %(prog)s data.jsonl converted_data.jsonl --prompt-file prompts.jsonl
  %(prog)s input.jsonl output.jsonl --prompt-file prompts.jsonl --verbose
        """
    )
    
    parser.add_argument(
        '--input_file',
        help='Path to the input JSONL file to convert',
        required=True
    )
    
    parser.add_argument(
        '--output_file',
        help='Path to the output JSONL file to create',
        required=True
    )
    
    parser.add_argument(
        '--prompt_file', '-p',
        dest='prompt_file',
        help='Optional path to a JSONL file containing prompts to include in the output'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress and error information'
    )
    
    return parser.parse_args()

def main():
    """Main function to handle command line arguments and run conversion."""
    
    args = parse_arguments()
    
    print(f"Converting JSONL format...")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.prompt_file:
        print(f"Prompt file: {args.prompt_file}")
    if args.verbose:
        print("Verbose mode: enabled")
    print("-" * 50)
    
    convert_jsonl_file(
        input_file=args.input_file,
        output_file=args.output_file,
        prompt_file=args.prompt_file,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
