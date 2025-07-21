import openai
import os
import json
import re
from typing import Dict, List, Any
from datetime import datetime

from models.schemas import ParsedQuery, EntityExtraction, DomainType

class LLMParser:
    def __init__(self):
        # Initialize OpenAI client with custom base URL
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1/")
        )
        
        # Configuration from environment
        self.max_chunks_per_query = int(os.getenv("MAX_CHUNKS_PER_QUERY", "3"))
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
        self.max_answer_tokens = int(os.getenv("MAX_ANSWER_TOKENS", "300"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Domain-specific prompts
        self.domain_prompts = {
            DomainType.INSURANCE: self._get_insurance_prompt(),
            DomainType.LEGAL: self._get_legal_prompt(),
            DomainType.HR: self._get_hr_prompt(),
            DomainType.COMPLIANCE: self._get_compliance_prompt()
        }
    
    async def generate_answer_from_chunks(self, question: str, relevant_chunks: List[str], domain: str = "insurance") -> Dict[str, Any]:
        """
        Generate answer from relevant chunks using efficient token management
        Only processes the most relevant chunks to avoid API exhaustion
        """
        try:
            # Limit the number of chunks and total context size
            max_chunks = self.max_chunks_per_query
            max_context_length = self.max_context_length
            
            # Select top chunks and limit context size
            selected_chunks = relevant_chunks[:max_chunks]
            context = ""
            
            for chunk in selected_chunks:
                if len(context) + len(chunk) > max_context_length:
                    # Truncate to fit within limit
                    remaining_space = max_context_length - len(context)
                    if remaining_space > 100:  # Only add if meaningful space left
                        context += chunk[:remaining_space] + "..."
                    break
                context += chunk + "\n\n"
            
            if not context.strip():
                return {
                    "answer": "No relevant information found in the document.",
                    "confidence": 0.1,
                    "reasoning": "No context available",
                    "sources_used": 0
                }
            
            # Create efficient prompt
            prompt = f"""Based on the following document context, answer the question concisely:

Context:
{context}

Question: {question}

Provide a direct answer based only on the given context. If the information is not in the context, say so."""

            # Call OpenAI with optimized parameters
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": f"You are a {domain} document analyst. Answer questions based only on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=self.max_answer_tokens,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on answer quality and context relevance
            confidence = self._calculate_answer_confidence(answer, context, question)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "reasoning": f"Answer derived from {len(selected_chunks)} document sections",
                "sources_used": len(selected_chunks),
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "reasoning": "Processing error occurred",
                "sources_used": 0
            }
    
    def _calculate_answer_confidence(self, answer: str, context: str, question: str) -> float:
        """Calculate confidence score for the generated answer"""
        confidence = 0.5  # Base confidence
        
        # Check if answer indicates uncertainty
        uncertainty_phrases = ["not mentioned", "not specified", "not available", "cannot determine"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        # Check if answer has specific details
        if any(char.isdigit() or char == '%' for char in answer):
            confidence += 0.2
        
        # Check context relevance
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        overlap = len(question_words.intersection(context_words))
        if overlap > 3:
            confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def parse_query(self, query: str, domain: str) -> Dict[str, Any]:
        """
        Parse natural language query and extract structured information
        Uses efficient token management to avoid API exhaustion
        """
        try:
            domain_enum = DomainType(domain)
            prompt = self.domain_prompts[domain_enum]
            
            # Create the full prompt - keep it concise to save tokens
            full_prompt = f"{prompt}\n\nUser Query: {query}\n\nProvide structured JSON response."
            
            # Call OpenAI API with optimized parameters
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant. Respond in JSON format only."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=800,  # Reduced token limit for query parsing
                top_p=0.9
            )
            
            # Parse the response
            result = self._parse_llm_response(response.choices[0].message.content, query, domain)
            
            return result
            
        except Exception as e:
            print(f"LLM parsing failed, using fallback: {e}")
            # Fallback parsing if LLM fails
            return self._fallback_parse(query, domain)
    
    def _parse_llm_response(self, llm_response: str, original_query: str, domain: str) -> Dict[str, Any]:
        """Parse LLM response and structure it"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                parsed_data = {}
            
            # Extract entities
            entities = []
            if "entities" in parsed_data:
                for entity_data in parsed_data["entities"]:
                    entities.append(EntityExtraction(
                        entity_type=entity_data.get("type", "unknown"),
                        value=entity_data.get("value", ""),
                        confidence=entity_data.get("confidence", 0.8),
                        position=(entity_data.get("start", 0), entity_data.get("end", 0))
                    ))
            
            return {
                "original_query": original_query,
                "processed_query": parsed_data.get("processed_query", original_query),
                "intent": parsed_data.get("intent", "information_retrieval"),
                "entities": entities,
                "query_type": parsed_data.get("query_type", "general"),
                "domain_specific_terms": parsed_data.get("domain_specific_terms", []),
                "confidence": parsed_data.get("confidence", 0.8)
            }
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_parse(original_query, domain)
    
    def _fallback_parse(self, query: str, domain: str) -> Dict[str, Any]:
        """Fallback parsing using rule-based approach"""
        # Basic entity extraction using regex patterns
        entities = []
        
        # Common patterns for different domains
        patterns = {
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            "amount": r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b',
            "policy_number": r'\b[A-Z]{2,4}\d{6,12}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(EntityExtraction(
                    entity_type=entity_type,
                    value=match.group(),
                    confidence=0.6,
                    position=(match.start(), match.end())
                ))
        
        # Determine intent based on keywords
        intent = "information_retrieval"
        if any(word in query.lower() for word in ["approve", "deny", "reject", "accept"]):
            intent = "decision_making"
        elif any(word in query.lower() for word in ["explain", "why", "reason", "justify"]):
            intent = "explanation"
        elif any(word in query.lower() for word in ["find", "search", "lookup", "retrieve"]):
            intent = "information_retrieval"
        
        return {
            "original_query": query,
            "processed_query": query.lower(),
            "intent": intent,
            "entities": entities,
            "query_type": "general",
            "domain_specific_terms": self._extract_domain_terms(query, domain),
            "confidence": 0.6
        }
    
    def _extract_domain_terms(self, query: str, domain: str) -> List[str]:
        """Extract domain-specific terms"""
        domain_vocabularies = {
            "insurance": ["policy", "premium", "claim", "deductible", "coverage", "underwriting", "risk", "liability"],
            "legal": ["contract", "clause", "agreement", "liability", "jurisdiction", "breach", "compliance", "statute"],
            "hr": ["employee", "benefits", "policy", "leave", "performance", "compensation", "workplace", "guidelines"],
            "compliance": ["regulation", "audit", "requirements", "standards", "violation", "procedure", "documentation"]
        }
        
        terms = []
        vocabulary = domain_vocabularies.get(domain, [])
        
        for term in vocabulary:
            if term.lower() in query.lower():
                terms.append(term)
        
        return terms
    
    def _get_insurance_prompt(self) -> str:
        return """
        You are an insurance document analysis expert. Analyze the user query and extract:
        1. Intent (claim_processing, policy_inquiry, risk_assessment, underwriting, etc.)
        2. Entities (policy numbers, claim amounts, dates, parties involved)
        3. Insurance-specific terms and concepts
        4. Query classification (eligibility, coverage, exclusions, etc.)
        
        Return a JSON object with: intent, entities, processed_query, query_type, domain_specific_terms, confidence
        """
    
    def _get_legal_prompt(self) -> str:
        return """
        You are a legal document analysis expert. Analyze the user query and extract:
        1. Intent (contract_review, clause_analysis, compliance_check, legal_research, etc.)
        2. Entities (contract parties, dates, amounts, legal references)
        3. Legal terms and concepts
        4. Query classification (liability, obligations, rights, etc.)
        
        Return a JSON object with: intent, entities, processed_query, query_type, domain_specific_terms, confidence
        """
    
    def _get_hr_prompt(self) -> str:
        return """
        You are an HR document analysis expert. Analyze the user query and extract:
        1. Intent (policy_lookup, benefits_inquiry, procedure_check, employee_rights, etc.)
        2. Entities (employee names, departments, dates, benefit types)
        3. HR-specific terms and concepts
        4. Query classification (policies, procedures, benefits, compliance, etc.)
        
        Return a JSON object with: intent, entities, processed_query, query_type, domain_specific_terms, confidence
        """
    
    def _get_compliance_prompt(self) -> str:
        return """
        You are a compliance document analysis expert. Analyze the user query and extract:
        1. Intent (regulation_check, audit_support, requirement_analysis, violation_assessment, etc.)
        2. Entities (regulation names, requirements, dates, compliance standards)
        3. Compliance-specific terms and concepts
        4. Query classification (requirements, procedures, standards, violations, etc.)
        
        Return a JSON object with: intent, entities, processed_query, query_type, domain_specific_terms, confidence
        """
