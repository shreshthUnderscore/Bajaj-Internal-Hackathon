import re
import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from models.schemas import ClauseMatch, ParsedQuery, EmbeddingResult, DomainType

class ClauseMatcher:
    def __init__(self):
        # Domain-specific clause patterns and vocabularies
        self.domain_patterns = {
            DomainType.INSURANCE: self._get_insurance_patterns(),
            DomainType.LEGAL: self._get_legal_patterns(),
            DomainType.HR: self._get_hr_patterns(),
            DomainType.COMPLIANCE: self._get_compliance_patterns()
        }
        
        # Initialize TF-IDF vectorizer for semantic matching
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        
        # Clause type classifications
        self.clause_types = {
            "liability": ["liability", "responsible", "liable", "fault", "damages"],
            "coverage": ["coverage", "covered", "include", "exclude", "benefit"],
            "exclusion": ["exclude", "not covered", "exception", "limitation"],
            "procedure": ["procedure", "process", "step", "requirement", "must"],
            "deadline": ["deadline", "time limit", "within", "before", "after"],
            "payment": ["payment", "pay", "fee", "cost", "charge", "premium"],
            "termination": ["terminate", "end", "expire", "cancel", "dissolution"],
            "confidentiality": ["confidential", "private", "non-disclosure", "secret"]
        }
    
    async def match_clauses(
        self, 
        parsed_query: Dict[str, Any], 
        search_results: List[EmbeddingResult], 
        domain: str
    ) -> List[ClauseMatch]:
        """
        Match and classify clauses based on query intent and domain
        """
        try:
            domain_enum = DomainType(domain)
            patterns = self.domain_patterns.get(domain_enum, {})
            
            matched_clauses = []
            
            for result in search_results:
                # Extract potential clauses from the text
                clauses = self._extract_clauses(result.text, patterns)
                
                for clause in clauses:
                    # Calculate semantic similarity
                    similarity = self._calculate_semantic_similarity(
                        parsed_query["processed_query"], 
                        clause["text"]
                    )
                    
                    if similarity > 0.3:  # Threshold for relevance
                        # Classify clause type
                        clause_type = self._classify_clause_type(clause["text"])
                        
                        clause_match = ClauseMatch(
                            clause_id=f"{result.chunk_id}_{clause['index']}",
                            clause_text=clause["text"],
                            similarity_score=similarity,
                            clause_type=clause_type,
                            document_source=result.document_id,
                            metadata={
                                "chunk_id": result.chunk_id,
                                "original_similarity": result.similarity_score,
                                "pattern_matched": clause.get("pattern", "semantic"),
                                "confidence": clause.get("confidence", similarity)
                            }
                        )
                        matched_clauses.append(clause_match)
            
            # Sort by similarity score and remove duplicates
            matched_clauses = self._deduplicate_clauses(matched_clauses)
            matched_clauses.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return matched_clauses[:20]  # Return top 20 matches
        
        except Exception as e:
            print(f"Error in clause matching: {e}")
            return []
    
    def _extract_clauses(self, text: str, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract clauses from text using patterns and rules"""
        clauses = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check for pattern matches
            pattern_matched = False
            for pattern_name, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        clauses.append({
                            "text": sentence,
                            "index": i,
                            "pattern": pattern_name,
                            "confidence": 0.8
                        })
                        pattern_matched = True
                        break
                if pattern_matched:
                    break
            
            # If no pattern matched, include potentially relevant sentences
            if not pattern_matched and self._is_potential_clause(sentence):
                clauses.append({
                    "text": sentence,
                    "index": i,
                    "pattern": "semantic",
                    "confidence": 0.6
                })
        
        return clauses
    
    def _is_potential_clause(self, sentence: str) -> bool:
        """Determine if a sentence could be a relevant clause"""
        # Look for legal/contractual language indicators
        indicators = [
            "shall", "must", "will", "agree", "subject to", "provided that",
            "in the event", "notwithstanding", "hereby", "whereas", "therefore",
            "liable", "responsible", "entitled", "obligation", "right", "duty"
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in indicators)
    
    def _calculate_semantic_similarity(self, query: str, clause: str) -> float:
        """Calculate semantic similarity between query and clause"""
        try:
            # Combine texts for TF-IDF fitting
            texts = [query, clause]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _classify_clause_type(self, clause_text: str) -> str:
        """Classify the type of clause"""
        clause_lower = clause_text.lower()
        
        best_match = "general"
        best_score = 0
        
        for clause_type, keywords in self.clause_types.items():
            score = sum(1 for keyword in keywords if keyword in clause_lower)
            if score > best_score:
                best_score = score
                best_match = clause_type
        
        return best_match
    
    def _deduplicate_clauses(self, clauses: List[ClauseMatch]) -> List[ClauseMatch]:
        """Remove duplicate or very similar clauses"""
        if not clauses:
            return clauses
        
        deduplicated = []
        for clause in clauses:
            is_duplicate = False
            for existing in deduplicated:
                # Check for high text similarity
                similarity = self._calculate_semantic_similarity(
                    clause.clause_text, 
                    existing.clause_text
                )
                if similarity > 0.9:  # Very high similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(clause)
        
        return deduplicated
    
    def _get_insurance_patterns(self) -> Dict[str, List[str]]:
        """Insurance-specific clause patterns"""
        return {
            "coverage": [
                r"covered\s+(?:under|by|for)",
                r"coverage\s+(?:includes|extends|applies)",
                r"insured\s+(?:for|against)",
                r"benefits\s+(?:include|cover)"
            ],
            "exclusion": [
                r"(?:not\s+)?(?:excluded|excepted)",
                r"does\s+not\s+cover",
                r"limitation\s+of\s+coverage",
                r"exclusions?\s+(?:include|are)"
            ],
            "liability": [
                r"liable\s+(?:for|to)",
                r"responsibility\s+(?:for|of)",
                r"damages\s+(?:caused|resulting)",
                r"fault\s+(?:of|by)"
            ],
            "premium": [
                r"premium\s+(?:payment|due)",
                r"cost\s+of\s+(?:coverage|insurance)",
                r"fee\s+(?:for|of)"
            ]
        }
    
    def _get_legal_patterns(self) -> Dict[str, List[str]]:
        """Legal document clause patterns"""
        return {
            "obligation": [
                r"shall\s+(?:be\s+)?(?:required|obligated)",
                r"must\s+(?:be\s+)?(?:provided|delivered|completed)",
                r"agrees?\s+to",
                r"undertakes?\s+to"
            ],
            "liability": [
                r"liable\s+(?:for|to|in)",
                r"responsible\s+(?:for|to)",
                r"indemnify\s+(?:and\s+)?hold\s+harmless",
                r"damages\s+(?:arising|resulting)"
            ],
            "termination": [
                r"terminate\s+(?:this|the)",
                r"expir(?:e|ation)\s+(?:of|on)",
                r"breach\s+of\s+(?:contract|agreement)",
                r"dissolution\s+of"
            ],
            "jurisdiction": [
                r"governed\s+by\s+(?:the\s+)?laws",
                r"jurisdiction\s+(?:of|in)",
                r"courts?\s+of",
                r"dispute\s+resolution"
            ]
        }
    
    def _get_hr_patterns(self) -> Dict[str, List[str]]:
        """HR policy clause patterns"""
        return {
            "benefits": [
                r"entitled\s+to\s+(?:receive|benefits)",
                r"benefits?\s+(?:include|comprise)",
                r"eligible\s+(?:for|to\s+receive)",
                r"vacation\s+(?:time|days|leave)"
            ],
            "procedures": [
                r"procedure\s+(?:for|to)",
                r"process\s+(?:includes|requires)",
                r"steps?\s+(?:to|for)",
                r"required\s+to\s+(?:follow|complete)"
            ],
            "conduct": [
                r"conduct\s+(?:standards|expectations)",
                r"behavior\s+(?:that|which)",
                r"prohibited\s+(?:activities|conduct)",
                r"disciplinary\s+(?:action|measures)"
            ],
            "leave": [
                r"leave\s+(?:of\s+absence|policy)",
                r"time\s+off",
                r"sick\s+(?:leave|days)",
                r"maternity\s+(?:leave|benefits)"
            ]
        }
    
    def _get_compliance_patterns(self) -> Dict[str, List[str]]:
        """Compliance regulation clause patterns"""
        return {
            "requirements": [
                r"requirement\s+(?:to|for|that)",
                r"must\s+(?:comply|adhere|conform)",
                r"mandatory\s+(?:that|to)",
                r"regulation\s+(?:requires|mandates)"
            ],
            "procedures": [
                r"procedure\s+(?:for|to\s+ensure)",
                r"process\s+(?:must|shall)",
                r"audit\s+(?:procedures|requirements)",
                r"documentation\s+(?:required|must)"
            ],
            "violations": [
                r"violation\s+(?:of|includes)",
                r"breach\s+of\s+(?:compliance|regulations)",
                r"non-compliance\s+(?:with|results)",
                r"penalty\s+(?:for|includes)"
            ],
            "standards": [
                r"standards?\s+(?:require|mandate)",
                r"criteria\s+(?:for|include)",
                r"benchmark\s+(?:for|of)",
                r"measures?\s+(?:to\s+ensure|for)"
            ]
        }
