"""
Enhanced Entity and Relation Extraction for CFO Contract Analytics
LLM-based and rule-based extraction for financial contracts
"""

import re
import openai
from typing import List, Tuple

import config
import utils

# Set OpenAI API key
openai.api_key = config.OPENAI_API_KEY

# ==================== LLM Communication ====================

def chat_strict(messages, temperature=0.1, timeout=45, model="gpt-4"):
    """Resilient LLM call with fallback support"""
    # Works for both new and legacy clients
    try:
        comp = openai.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=float(temperature), 
            timeout=timeout
        )
        return (comp.choices[0].message.content or "").strip()
    except Exception:
        pass
    try:
        comp = openai.ChatCompletion.create(
            model=model, 
            messages=messages, 
            temperature=float(temperature), 
            timeout=timeout
        )
        return (comp.choices[0].message.content or "").strip()
    except Exception:
        return ""

# ==================== Entity and Relation Extraction ====================

def extract_entities_relations(text: str) -> List[Tuple[str, str, str, str]]:
    """
    Extract financial entity-relation-entity triples from contract text
    Returns list of (head, relation, tail, evidence) tuples
    """
    content = text[:20000]  # Limit content size
    chunks = utils.chunk_text(content, max_chars=config.CHUNK_MAX_CHARS)
    triples_with_source = []

    base_prompt = (
        'Extract financial entity–relation–entity triples from the text. '
        'Return STRICT JSON as: '
        '{"triples":[{"head":"...", "relation":"...", "tail":"...", "evidence":"original sentence"}]}.\n'
        'Guidelines:\n'
        '- Use short, canonical entity names.\n'
        '- Relations must be concise verbs/phrases (e.s., "recognised_under","audited_by","governed_by").\n'
        '- "evidence" must be the exact sentence containing the relation.\n'
        '- Skip trivial or ambiguous lines.\n'
    )
    
    if bool(config.STRICT_PROMPT):
        base_prompt += (
            '- Reject edges if the relation verb is missing.\n'
            '- If evidence is unavailable, return an empty "triples" array.\n'
            '- Do not infer beyond text; avoid co-occurrence-only edges.\n'
        )
    base_prompt += 'Text:\n'

    for ch in chunks:
        msg = [{"role": "user", "content": base_prompt + ch}]
        try:
            raw = chat_strict(msg, temperature=config.EXTRACT_TEMP, timeout=45, model="gpt-4")
            parsed = utils.parse_json_or_lines(raw)
            triples_with_source.extend(parsed)
        except Exception:
            pass

    # Fallback to rule-based extraction if no triples found
    if not triples_with_source:
        triples_with_source = utils.rule_based_triples(content)

    # Further fallback to co-occurrence if enabled
    if not triples_with_source and bool(config.ENABLE_FALLBACK_COOCC):
        for s in re.split(r'(?<=[.!?])\s+', content):
            if s.strip():
                triples_with_source.extend(utils.fallback_cooccurrence(s))

    # Clean and validate triples
    cleaned, seen = [], set()
    for a, r, b, ev in triples_with_source:
        a = utils.canon_entity(a)
        b = utils.canon_entity(b)
        r_norm = utils.norm_rel(r)
        ev = (ev or "").strip()
        
        if not a or not b or a.lower() == b.lower():
            continue
        if not utils.relation_allowed(r_norm):
            continue
        if (bool(config.SKIP_EDGES_NO_EVID or config.REQUIRE_EVIDENCE)) and not ev:
            continue
            
        key = (a, r_norm, b, ev)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    
    return cleaned

# ==================== CFO-Specific Extraction Patterns ====================

def extract_financial_terms(text: str) -> dict:
    """
    Extract specific financial terms and patterns from contract text
    Focus on CFO-relevant metrics and clauses
    """
    patterns = {
        'contract_value': [
            r'total\s+(?:contract\s+)?(?:value|amount|fee|cost)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'contract\s+(?:value|amount)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'agreement\s+(?:value|amount)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)'
        ],
        'annual_commitment': [
            r'annual\s+(?:commitment|obligation|fee)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'yearly\s+(?:cost|fee|payment)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)'
        ],
        'payment_terms': [
            r'payment\s+(?:terms|period|schedule)[:\s]*([^\n]{20,100})',
            r'(?:net\s+\d+|milestone[^n]|payable)[^,]*,?\s*([^\n]{10,80})'
        ],
        'escalation': [
            r'escalation[^,]*(?:%|percent|rate)[^,]*(?:\d+(?:\.\d+)?%?)',
            r'annual\s+increase[^,]*(?:\d+(?:\.\d+)?%?)'
        ],
        'sla_terms': [
            r'sla[^,]*(?:uptime|availability)[^,]*\d+(?:\.\d+)?%',
            r'(?:99\.\d+|uptime)[^,]*\d+(?:\.\d+)?%'
        ],
        'termination_clauses': [
            r'termination[^,]*(?:[^,]*,?\s*){10,40}[^,]*(?:breach|notice|cause)',
            r'(?:end|expire|terminate)[^,]*(?:notice|cure|breach)'
        ],
        'compliance_requirements': [
            r'(?:gdpr|sox|ifrs|asc|gaap|iso|pci|soc2)[^,]*(?:compliant|required)',
            r'compliance[^,]*with[^,]*(?:standards|regulations)'
        ],
        'renewal_terms': [
            r'renewal[^,]*(?:automatic|manual|optional)[^,]*(?:[^,]*,?\s*){5,20}',
            r'(?:auto.*renew|renewal.*auto)'
        ]
    }
    
    extracted = {}
    
    for term_type, regex_list in patterns.items():
        extracted[term_type] = []
        for regex in regex_list:
            matches = re.findall(regex, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                if match.strip() and len(match.strip()) > 5:
                    extracted[term_type].append(match.strip())
    
    return extracted

# ==================== Extraction Quality Metrics ====================

def calculate_extraction_quality(extraction_output: List[Tuple]) -> dict:
    """
    Calculate quality metrics for extraction results
    """
    if not extraction_output:
        return {"quality_score": 0.0, "evidence_rate": 0.0, "relation_diversity": 0.0}
    
    total_triples = len(extraction_output)
    evidence_count = sum(1 for _, _, _, ev in extraction_output if ev.strip())
    unique_relations = len(set(rel for _, rel, _, _ in extraction_output))
    
    evidence_rate = evidence_count / total_triples if total_triples > 0 else 0.0
    relation_diversity = unique_relations / total_triples if total_triples > 0 else 0.0
    
    # Weighted quality score
    quality_score = (0.4 * evidence_rate + 0.3 * relation_diversity + 
                    0.3 * min(1.0, evidence_count / 10))  # Bonus for reasonable extraction count
    
    return {
        "quality_score": round(quality_score, 3),
        "evidence_rate": round(evidence_rate, 3),
        "relation_diversity": round(relation_diversity, 3),
        "total_triples": total_triples,
        "evidence_count": evidence_count,
        "unique_relations": unique_relations
    }





