import os
import re
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
    - Removes excessive whitespace (multiple spaces, tabs, newlines).
    - Strips leading/trailing whitespace from lines.
    - Removes common non-essential characters (e.g., form feed characters).
    """
    if not isinstance(text, str):
        return "" # Ensure input is a string

    # Remove form feed characters
    text = text.replace('\x0c', '')
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace from each line
    text = '\n'.join([line.strip() for line in text.split('\n')]).strip()
    logging.debug("Text cleaned successfully.")
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Chunks a given text into smaller pieces using RecursiveCharacterTextSplitter.
    This splitter tries to split by paragraphs, then sentences, then words, then characters.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use character length
        add_start_index=True, # Add start index to metadata
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Text chunked into {len(chunks)} pieces (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")
    return chunks

def extract_metadata(document_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts basic metadata from the document content.
    This is a placeholder for more advanced, domain-specific metadata extraction.
    For policy/contract data, this would involve regex or NLP models.
    """
    metadata = {
        "document_id": document_content.get("document_id"),
        "file_path": document_content.get("file_path"),
        "document_type": document_content.get("document_type"),
        "source": os.path.basename(document_content.get("file_path", "unknown_source")),
        "domain": "general" # Default, can be updated based on content or file path
    }

    text_content = document_content.get("text_content", "")

    # --- Placeholder for domain-specific metadata extraction ---
    # Example for Insurance:
    if "insurance" in text_content.lower() or "policy" in text_content.lower():
        metadata["domain"] = "Insurance"
        # Example: Try to find a policy number (very basic regex)
        policy_number_match = re.search(r'(Policy|Pol\.)\s*No\.?\s*[:\-]?\s*([A-Z0-9]{5,})', text_content, re.IGNORECASE)
        if policy_number_match:
            metadata["policy_number"] = policy_number_match.group(2)

    # Example for Legal:
    if "contract" in text_content.lower() or "agreement" in text_content.lower() or "legal" in text_content.lower():
        metadata["domain"] = "Legal"
        # Example: Try to find contract date
        contract_date_match = re.search(r'(Dated|Date of this Agreement)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})', text_content, re.IGNORECASE)
        if contract_date_match:
            metadata["contract_date"] = contract_date_match.group(2)

    # Example for HR:
    if "employee handbook" in text_content.lower() or "hr policy" in text_content.lower():
        metadata["domain"] = "HR"
        # Example: Try to find department
        department_match = re.search(r'(Department|Dept\.)\s*[:\-]?\s*([A-Za-z\s]+)', text_content)
        if department_match:
            metadata["department"] = department_match.group(2).strip()

    # Example for Compliance:
    if "compliance" in text_content.lower() or "regulation" in text_content.lower() or "gdpr" in text_content.lower():
        metadata["domain"] = "Compliance"
        # Example: Try to find regulation ID
        regulation_id_match = re.search(r'(GDPR|HIPAA|ISO)\s*(\d{4,})', text_content, re.IGNORECASE)
        if regulation_id_match:
            metadata["regulation_id"] = regulation_id_match.group(0) # e.g., "GDPR 2016"

    logging.info(f"Extracted metadata for {metadata.get('document_id')}: {metadata}")
    return metadata

def process_document(document_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes a single document: cleans text, extracts metadata, and chunks the text.
    Returns a list of dictionaries, where each dictionary represents a chunk
    with its content and associated metadata.
    """
    original_text = document_content.get("text_content", "")
    document_id = document_content.get("document_id", "unknown_doc")

    # 1. Clean Text
    cleaned_text = clean_text(original_text)
    if not cleaned_text:
        logging.warning(f"No text content to process for document: {document_id}")
        return []

    # 2. Extract Metadata
    base_metadata = extract_metadata(document_content)

    # 3. Chunk Text
    chunks = chunk_text(cleaned_text)

    processed_chunks = []
    for i, chunk_text_content in enumerate(chunks):
        chunk_metadata = base_metadata.copy() # Start with base document metadata
        chunk_metadata["chunk_id"] = f"{document_id}_chunk_{i}"
        chunk_metadata["chunk_index"] = i
        # Add start index if available from RecursiveCharacterTextSplitter
        # Note: Langchain's split_text returns list of strings, not Document objects
        # For start_index, you'd typically use text_splitter.create_documents(text)
        # For simplicity here, we'll assume the chunk_index is sufficient for now.
        # If precise character start index is needed, we'd adjust `chunk_text` function
        # to return Langchain Document objects.

        processed_chunks.append({
            "content": chunk_text_content,
            "metadata": chunk_metadata
        })
    logging.info(f"Document {document_id} processed into {len(processed_chunks)} chunks.")
    return processed_chunks

if __name__ == "__main__":
    # Example usage with a dummy document dictionary
    print("--- Testing Document Preprocessing and Chunking ---")

    # Simulate a document loaded by document_loader.py
    sample_doc_data = {
        "file_path": "/path/to/my_policy_document.docx",
        "document_type": "docx",
        "text_content": """
        This is a sample policy document.

        Section 1: General Provisions.
        This policy (Policy No. ABC12345) is effective from 01/01/2025.
        It covers accidental damage.

        Section 2: Exclusions.
        Damage due to intentional acts is excluded.
        The deductible for accidental damage is $500.

        This document also mentions compliance with GDPR.
        """ * 5 # Make it longer to ensure chunking occurs
    }

    processed_chunks = process_document(sample_doc_data)

    print(f"\nTotal processed chunks: {len(processed_chunks)}")
    if processed_chunks:
        print("\nFirst chunk content and metadata:")
        print(f"Content: {processed_chunks[0]['content']}")
        print(f"Metadata: {processed_chunks[0]['metadata']}")

        if len(processed_chunks) > 1:
            print("\nSecond chunk content and metadata:")
            print(f"Content: {processed_chunks[1]['content']}")
            print(f"Metadata: {processed_chunks[1]['metadata']}")

    # Test with an empty document
    empty_doc_data = {
        "file_path": "/path/to/empty.txt",
        "document_type": "txt",
        "text_content": ""
    }
    empty_chunks = process_document(empty_doc_data)
    print(f"\nTotal chunks for empty document: {len(empty_chunks)}")

    # Test with a very short document
    short_doc_data = {
        "file_path": "/path/to/short.txt",
        "document_type": "txt",
        "text_content": "This is a very short text."
    }
    short_chunks = process_document(short_doc_data)
    print(f"\nTotal chunks for short document: {len(short_chunks)}")
    if short_chunks:
        print(f"Content: {short_chunks[0]['content']}")
        print(f"Metadata: {short_chunks[0]['metadata']}")

