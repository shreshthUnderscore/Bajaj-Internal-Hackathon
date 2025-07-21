import json
import os
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from collections import defaultdict

from models.schemas import (
    ClauseMatch, MatchedDocument, DecisionResult, 
    SourceReference, LogicEvaluationResult, DomainType
)

class LogicEvaluator:
    def __init__(self):
        # Decision rules for different domains
        self.decision_rules = {
            DomainType.INSURANCE: self._get_insurance_rules(),
            DomainType.LEGAL: self._get_legal_rules(),
            DomainType.HR: self._get_hr_rules(),
            DomainType.COMPLIANCE: self._get_compliance_rules()
        }
        
        # Storage for decision history
        self.decision_history = {}
        self.history_file = "decision_history.json"
        self.load_history()
    
    async def evaluate(
        self, 
        parsed_query: Dict[str, Any], 
        matched_clauses: List[ClauseMatch], 
        domain: str
    ) -> Dict[str, Any]:
        """
        Perform logic evaluation and decision processing
        """
        start_time = datetime.now()
        
        try:
            domain_enum = DomainType(domain)
            rules = self.decision_rules.get(domain_enum, {})
            
            # Step 1: Analyze matched clauses
            clause_analysis = self._analyze_clauses(matched_clauses, parsed_query["intent"])
            
            # Step 2: Group documents and calculate relevance
            document_groups = self._group_by_document(matched_clauses)
            matched_documents = self._create_matched_documents(document_groups)
            
            # Step 3: Apply decision logic
            decision_result = self._apply_decision_logic(
                parsed_query, 
                clause_analysis, 
                matched_documents, 
                rules
            )
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(
                parsed_query, 
                clause_analysis, 
                matched_documents
            )
            
            # Step 5: Generate explanation
            explanation = self._generate_explanation(
                parsed_query, 
                clause_analysis, 
                decision_result, 
                confidence_score
            )
            
            # Step 6: Create source references
            source_references = self._create_source_references(matched_clauses)
            
            # Step 7: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 8: Store decision for future reference
            decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            decision_record = {
                "decision_id": decision_id,
                "query": parsed_query["original_query"],
                "domain": domain,
                "decision": decision_result.dict() if hasattr(decision_result, 'dict') else decision_result,
                "confidence": confidence_score,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time
            }
            
            self.decision_history[decision_id] = decision_record
            self.save_history()
            
            return {
                "matched_documents": matched_documents,
                "decision": decision_result,
                "confidence_score": confidence_score,
                "explanation": explanation,
                "source_references": source_references,
                "processing_time": processing_time,
                "intermediate_steps": [
                    {"step": "clause_analysis", "result": clause_analysis},
                    {"step": "document_grouping", "count": len(document_groups)},
                    {"step": "decision_logic", "applied_rules": len(rules)},
                    {"step": "confidence_calculation", "score": confidence_score}
                ]
            }
        
        except Exception as e:
            print(f"Error in logic evaluation: {e}")
            return self._create_error_response(str(e))
    
    def _analyze_clauses(self, matched_clauses: List[ClauseMatch], intent: str) -> Dict[str, Any]:
        """Analyze matched clauses for patterns and insights"""
        analysis = {
            "total_clauses": len(matched_clauses),
            "clause_types": defaultdict(int),
            "avg_similarity": 0,
            "max_similarity": 0,
            "high_confidence_clauses": [],
            "coverage_analysis": {},
            "conflict_analysis": []
        }
        
        if not matched_clauses:
            return analysis
        
        # Basic statistics
        similarities = [clause.similarity_score for clause in matched_clauses]
        analysis["avg_similarity"] = np.mean(similarities)
        analysis["max_similarity"] = np.max(similarities)
        
        # Clause type distribution
        for clause in matched_clauses:
            analysis["clause_types"][clause.clause_type] += 1
        
        # High confidence clauses (top 20% by similarity)
        threshold = np.percentile(similarities, 80) if len(similarities) > 5 else 0.5
        analysis["high_confidence_clauses"] = [
            clause for clause in matched_clauses 
            if clause.similarity_score >= threshold
        ]
        
        # Intent-specific analysis
        if intent == "coverage":
            analysis["coverage_analysis"] = self._analyze_coverage(matched_clauses)
        elif intent == "liability":
            analysis["coverage_analysis"] = self._analyze_liability(matched_clauses)
        elif intent == "compliance":
            analysis["coverage_analysis"] = self._analyze_compliance(matched_clauses)
        
        # Conflict detection
        analysis["conflict_analysis"] = self._detect_conflicts(matched_clauses)
        
        return analysis
    
    def _group_by_document(self, matched_clauses: List[ClauseMatch]) -> Dict[str, List[ClauseMatch]]:
        """Group clauses by document"""
        groups = defaultdict(list)
        for clause in matched_clauses:
            groups[clause.document_source].append(clause)
        return dict(groups)
    
    def _create_matched_documents(self, document_groups: Dict[str, List[ClauseMatch]]) -> List[MatchedDocument]:
        """Create MatchedDocument objects from clause groups"""
        matched_docs = []
        
        for doc_id, clauses in document_groups.items():
            # Calculate overall relevance score for the document
            relevance_score = np.mean([clause.similarity_score for clause in clauses])
            
            # Extract matched sections
            matched_sections = [clause.clause_text[:100] + "..." for clause in clauses[:3]]
            
            matched_doc = MatchedDocument(
                document_id=doc_id,
                filename=f"document_{doc_id}",  # This should be fetched from metadata
                relevance_score=relevance_score,
                matched_sections=matched_sections
            )
            matched_docs.append(matched_doc)
        
        # Sort by relevance
        matched_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        return matched_docs
    
    def _apply_decision_logic(
        self, 
        parsed_query: Dict[str, Any], 
        clause_analysis: Dict[str, Any], 
        matched_documents: List[MatchedDocument], 
        rules: Dict[str, Any]
    ) -> DecisionResult:
        """Apply domain-specific decision logic"""
        
        intent = parsed_query["intent"]
        confidence_threshold = rules.get("confidence_threshold", 0.7)
        
        # Default decision structure
        decision = {
            "decision_type": intent,
            "outcome": "inconclusive",
            "reasoning": [],
            "risk_level": "medium",
            "recommendations": []
        }
        
        # Apply intent-specific logic
        if intent in ["coverage", "eligibility"]:
            decision = self._evaluate_coverage_decision(clause_analysis, matched_documents, rules)
        elif intent in ["liability", "responsibility"]:
            decision = self._evaluate_liability_decision(clause_analysis, matched_documents, rules)
        elif intent in ["compliance", "requirement"]:
            decision = self._evaluate_compliance_decision(clause_analysis, matched_documents, rules)
        elif intent in ["claim_processing", "approval"]:
            decision = self._evaluate_approval_decision(clause_analysis, matched_documents, rules)
        else:
            decision = self._evaluate_general_decision(clause_analysis, matched_documents, rules)
        
        return DecisionResult(**decision)
    
    def _evaluate_coverage_decision(self, analysis: Dict, documents: List, rules: Dict) -> Dict:
        """Evaluate coverage-related decisions"""
        decision = {
            "decision_type": "coverage_evaluation",
            "outcome": "covered",
            "reasoning": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        # Check coverage clauses
        coverage_clauses = analysis["clause_types"].get("coverage", 0)
        exclusion_clauses = analysis["clause_types"].get("exclusion", 0)
        
        if coverage_clauses > exclusion_clauses:
            decision["outcome"] = "covered"
            decision["reasoning"].append(f"Found {coverage_clauses} coverage clauses vs {exclusion_clauses} exclusion clauses")
        elif exclusion_clauses > coverage_clauses:
            decision["outcome"] = "not_covered"
            decision["reasoning"].append(f"Found {exclusion_clauses} exclusion clauses vs {coverage_clauses} coverage clauses")
            decision["risk_level"] = "high"
        else:
            decision["outcome"] = "partial_coverage"
            decision["reasoning"].append("Mixed coverage and exclusion clauses found")
            decision["risk_level"] = "medium"
        
        # Add confidence-based reasoning
        if analysis["avg_similarity"] < 0.5:
            decision["recommendations"].append("Low similarity scores suggest uncertain coverage determination")
        
        return decision
    
    def _evaluate_liability_decision(self, analysis: Dict, documents: List, rules: Dict) -> Dict:
        """Evaluate liability-related decisions"""
        decision = {
            "decision_type": "liability_evaluation",
            "outcome": "liable",
            "reasoning": [],
            "risk_level": "medium",
            "recommendations": []
        }
        
        liability_clauses = analysis["clause_types"].get("liability", 0)
        
        if liability_clauses > 2:
            decision["outcome"] = "liable"
            decision["risk_level"] = "high"
            decision["reasoning"].append(f"Multiple liability clauses found ({liability_clauses})")
        elif liability_clauses == 0:
            decision["outcome"] = "not_liable"
            decision["risk_level"] = "low"
            decision["reasoning"].append("No specific liability clauses found")
        else:
            decision["outcome"] = "limited_liability"
            decision["reasoning"].append("Limited liability clauses identified")
        
        return decision
    
    def _evaluate_compliance_decision(self, analysis: Dict, documents: List, rules: Dict) -> Dict:
        """Evaluate compliance-related decisions"""
        decision = {
            "decision_type": "compliance_evaluation",
            "outcome": "compliant",
            "reasoning": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        requirement_clauses = analysis["clause_types"].get("requirements", 0)
        violation_clauses = analysis["clause_types"].get("violations", 0)
        
        if violation_clauses > 0:
            decision["outcome"] = "non_compliant"
            decision["risk_level"] = "high"
            decision["reasoning"].append(f"Found {violation_clauses} potential violation clauses")
        elif requirement_clauses > 0:
            decision["outcome"] = "requires_review"
            decision["risk_level"] = "medium"
            decision["reasoning"].append(f"Found {requirement_clauses} requirement clauses needing review")
        else:
            decision["reasoning"].append("No compliance issues identified")
        
        return decision
    
    def _evaluate_approval_decision(self, analysis: Dict, documents: List, rules: Dict) -> Dict:
        """Evaluate approval-related decisions"""
        decision = {
            "decision_type": "approval_evaluation",
            "outcome": "approved",
            "reasoning": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        # Use average similarity as approval metric
        avg_sim = analysis["avg_similarity"]
        
        if avg_sim > 0.8:
            decision["outcome"] = "approved"
            decision["reasoning"].append("High similarity with policy documents")
        elif avg_sim > 0.6:
            decision["outcome"] = "conditional_approval"
            decision["risk_level"] = "medium"
            decision["reasoning"].append("Moderate similarity - conditional approval recommended")
        else:
            decision["outcome"] = "denied"
            decision["risk_level"] = "high"
            decision["reasoning"].append("Low similarity with policy requirements")
        
        return decision
    
    def _evaluate_general_decision(self, analysis: Dict, documents: List, rules: Dict) -> Dict:
        """Evaluate general decisions"""
        decision = {
            "decision_type": "general_evaluation",
            "outcome": "informational",
            "reasoning": [],
            "risk_level": "low",
            "recommendations": []
        }
        
        if analysis["total_clauses"] > 0:
            decision["reasoning"].append(f"Found {analysis['total_clauses']} relevant clauses")
            decision["recommendations"].append("Review matched clauses for detailed information")
        else:
            decision["reasoning"].append("No specific clauses matched the query")
            decision["recommendations"].append("Consider refining the query or checking additional documents")
        
        return decision
    
    def _calculate_confidence(
        self, 
        parsed_query: Dict, 
        clause_analysis: Dict, 
        matched_documents: List
    ) -> float:
        """Calculate overall confidence score"""
        factors = []
        
        # Query parsing confidence
        factors.append(parsed_query.get("confidence", 0.5))
        
        # Clause similarity scores
        if clause_analysis["total_clauses"] > 0:
            factors.append(clause_analysis["avg_similarity"])
        
        # Document relevance
        if matched_documents:
            avg_doc_relevance = np.mean([doc.relevance_score for doc in matched_documents])
            factors.append(avg_doc_relevance)
        
        # Number of matching documents (more is better, up to a point)
        doc_count_factor = min(len(matched_documents) / 5.0, 1.0)
        factors.append(doc_count_factor)
        
        # Calculate weighted average
        if factors:
            return float(np.mean(factors))
        else:
            return 0.5
    
    def _generate_explanation(
        self, 
        parsed_query: Dict, 
        clause_analysis: Dict, 
        decision_result: DecisionResult, 
        confidence_score: float
    ) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Query interpretation
        explanation_parts.append(f"Based on your query about '{parsed_query['original_query']}', ")
        explanation_parts.append(f"I identified this as a {parsed_query['intent']} request. ")
        
        # Analysis summary
        if clause_analysis["total_clauses"] > 0:
            explanation_parts.append(f"I found {clause_analysis['total_clauses']} relevant clauses ")
            explanation_parts.append(f"with an average similarity of {clause_analysis['avg_similarity']:.2f}. ")
        
        # Decision explanation
        explanation_parts.append(f"The decision outcome is '{decision_result.outcome}' ")
        explanation_parts.append(f"based on the following reasoning: {'; '.join(decision_result.reasoning)}. ")
        
        # Confidence explanation
        if confidence_score > 0.8:
            explanation_parts.append("I have high confidence in this decision. ")
        elif confidence_score > 0.6:
            explanation_parts.append("I have moderate confidence in this decision. ")
        else:
            explanation_parts.append("This decision has lower confidence and should be reviewed. ")
        
        # Risk assessment
        if decision_result.risk_level == "high":
            explanation_parts.append("This situation presents high risk and requires careful consideration. ")
        elif decision_result.risk_level == "medium":
            explanation_parts.append("This situation presents moderate risk. ")
        
        return "".join(explanation_parts)
    
    def _create_source_references(self, matched_clauses: List[ClauseMatch]) -> List[SourceReference]:
        """Create source references from matched clauses"""
        references = []
        
        for clause in matched_clauses[:10]:  # Top 10 references
            reference = SourceReference(
                document_id=clause.document_source,
                document_name=f"Document {clause.document_source}",
                section=clause.clause_type,
                confidence=clause.similarity_score,
                excerpt=clause.clause_text[:200] + "..." if len(clause.clause_text) > 200 else clause.clause_text
            )
            references.append(reference)
        
        return references
    
    def _analyze_coverage(self, clauses: List[ClauseMatch]) -> Dict[str, Any]:
        """Analyze coverage-specific patterns"""
        coverage_types = defaultdict(int)
        for clause in clauses:
            if "coverage" in clause.clause_text.lower():
                coverage_types["covered"] += 1
            elif "exclusion" in clause.clause_text.lower():
                coverage_types["excluded"] += 1
        
        return {"coverage_breakdown": dict(coverage_types)}
    
    def _analyze_liability(self, clauses: List[ClauseMatch]) -> Dict[str, Any]:
        """Analyze liability-specific patterns"""
        liability_indicators = defaultdict(int)
        for clause in clauses:
            text_lower = clause.clause_text.lower()
            if "liable" in text_lower:
                liability_indicators["direct_liability"] += 1
            elif "responsible" in text_lower:
                liability_indicators["responsibility"] += 1
            elif "indemnify" in text_lower:
                liability_indicators["indemnification"] += 1
        
        return {"liability_breakdown": dict(liability_indicators)}
    
    def _analyze_compliance(self, clauses: List[ClauseMatch]) -> Dict[str, Any]:
        """Analyze compliance-specific patterns"""
        compliance_aspects = defaultdict(int)
        for clause in clauses:
            text_lower = clause.clause_text.lower()
            if "requirement" in text_lower:
                compliance_aspects["requirements"] += 1
            elif "violation" in text_lower:
                compliance_aspects["violations"] += 1
            elif "standard" in text_lower:
                compliance_aspects["standards"] += 1
        
        return {"compliance_breakdown": dict(compliance_aspects)}
    
    def _detect_conflicts(self, clauses: List[ClauseMatch]) -> List[Dict[str, Any]]:
        """Detect potential conflicts between clauses"""
        conflicts = []
        
        # Simple conflict detection based on contradictory terms
        positive_terms = ["covered", "included", "entitled", "shall"]
        negative_terms = ["excluded", "not covered", "shall not", "prohibited"]
        
        positive_clauses = []
        negative_clauses = []
        
        for clause in clauses:
            text_lower = clause.clause_text.lower()
            if any(term in text_lower for term in positive_terms):
                positive_clauses.append(clause)
            elif any(term in text_lower for term in negative_terms):
                negative_clauses.append(clause)
        
        # If we have both positive and negative clauses, there might be conflicts
        if positive_clauses and negative_clauses:
            conflicts.append({
                "type": "coverage_conflict",
                "description": "Found both inclusive and exclusive clauses",
                "positive_count": len(positive_clauses),
                "negative_count": len(negative_clauses)
            })
        
        return conflicts
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "matched_documents": [],
            "decision": DecisionResult(
                decision_type="error",
                outcome="error",
                reasoning=[f"Error in processing: {error_message}"]
            ),
            "confidence_score": 0.0,
            "explanation": f"An error occurred during processing: {error_message}",
            "source_references": [],
            "processing_time": 0.0,
            "intermediate_steps": []
        }
    
    async def get_detailed_explanation(self, query_id: str) -> Dict[str, Any]:
        """Get detailed explanation for a specific decision"""
        if query_id in self.decision_history:
            return self.decision_history[query_id]
        else:
            return {"error": "Decision not found"}
    
    def load_history(self):
        """Load decision history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.decision_history = json.load(f)
        except Exception as e:
            print(f"Error loading decision history: {e}")
            self.decision_history = {}
    
    def save_history(self):
        """Save decision history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.decision_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving decision history: {e}")
    
    def _get_insurance_rules(self) -> Dict[str, Any]:
        """Insurance-specific decision rules"""
        return {
            "confidence_threshold": 0.7,
            "coverage_threshold": 0.8,
            "exclusion_weight": 1.5,
            "risk_factors": ["high_value", "multiple_claims", "previous_violations"]
        }
    
    def _get_legal_rules(self) -> Dict[str, Any]:
        """Legal document decision rules"""
        return {
            "confidence_threshold": 0.8,
            "liability_threshold": 0.75,
            "contract_weight": 1.2,
            "jurisdiction_factors": ["governing_law", "dispute_resolution"]
        }
    
    def _get_hr_rules(self) -> Dict[str, Any]:
        """HR policy decision rules"""
        return {
            "confidence_threshold": 0.6,
            "policy_threshold": 0.7,
            "employee_weight": 1.0,
            "compliance_factors": ["equal_opportunity", "safety_requirements"]
        }
    
    def _get_compliance_rules(self) -> Dict[str, Any]:
        """Compliance regulation decision rules"""
        return {
            "confidence_threshold": 0.9,
            "requirement_threshold": 0.85,
            "violation_weight": 2.0,
            "audit_factors": ["documentation", "procedures", "training"]
        }
