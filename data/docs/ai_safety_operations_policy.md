# Sentinel AI — AI Safety Operations Policies

## Overview

This document outlines the operational policies and compliance requirements for
AI-powered systems operating within the enterprise environment. All AI inference
systems must adhere to these policies.

---

## Section 1: Data Governance

### 1.1 Data Classification

All data processed by AI systems must be classified before analysis:

| Classification | Examples | Handling Requirement |
|----------------|----------|----------------------|
| **CONFIDENTIAL** | PII, financial records, trade secrets | Local pipeline ONLY |
| **INTERNAL** | Operations data, process metrics | Local pipeline preferred |
| **PUBLIC** | Marketing materials, public documents | Cloud pipeline allowed |

### 1.2 PII Detection Requirements

Sensitive Personally Identifiable Information (PII) that triggers mandatory local routing:
- Face images or videos containing identifiable individuals
- Government ID documents (passports, driver's licenses)
- Medical records and imagery
- Financial statements and account numbers
- Internal employee records

### 1.3 Audit Trail Requirements

Every AI inference request must generate an audit log entry containing:
- Request timestamp (ISO 8601)
- Data classification label
- Routing decision (local/cloud)
- Model used for inference
- Anonymized query hash (for compliance, not for model training)
- Processing time and token counts

---

## Section 2: Model Usage Policies

### 2.1 Approved Models — Local (On-Premise)

The following models are approved for processing CONFIDENTIAL data:
- LLaVA 1.6 (any size) — via Ollama
- Florence-2 — via HuggingFace Transformers
- Phi-3.5 Vision — via HuggingFace Transformers
- Moondream2 — via HuggingFace Transformers
- Llama 3.1 (any size) — via Ollama
- Mistral 7B — via Ollama

### 2.2 Cloud Models — Restricted

Cloud models (GPT-4V, Gemini Vision, Claude) may ONLY process PUBLIC or approved INTERNAL data:
- Must have explicit approval from Data Protection Officer (DPO)
- All cloud API calls must be logged
- Data minimization: send only what is necessary

### 2.3 Model Version Control

- All production model versions must be recorded in the Model Registry
- Model updates require security review before deployment
- Rollback procedures must be tested quarterly

---

## Section 3: System Requirements

### 3.1 Uptime and SLA

| Environment | Uptime Target | RTO | RPO |
|-------------|---------------|-----|-----|
| Production | 99.5% | 4 hours | 1 hour |
| Staging | 95% | 8 hours | 4 hours |
| Development | Best effort | N/A | N/A |

### 3.2 Performance Thresholds

Alert thresholds that trigger the on-call rotation:
- Vision inference latency > 10 seconds for >5% of requests (15-min window)
- LLM generation latency > 30 seconds for >10% of requests
- Error rate > 5% over any 5-minute window
- Memory usage > 90% of available RAM

### 3.3 Incident Response

1. **P0 (Critical)**: AI system fully down — page on-call immediately
2. **P1 (High)**: Degraded performance or data breach — respond within 1 hour
3. **P2 (Medium)**: Single model failure (fallback available) — respond within 4 hours
4. **P3 (Low)**: Performance degradation within SLA — respond within 24 hours

---

## Section 4: Responsible AI Guidelines

### 4.1 Bias and Fairness

- Regular bias assessments must be performed on production models (quarterly)
- Demographic parity evaluations required for any model affecting personnel decisions
- Bias findings must be reported to the AI Ethics Committee within 30 days

### 4.2 Explainability

- All AI-generated decisions with business impact must include a confidence score
- Decisions affecting individuals must be explainable in plain language on request
- Black-box model outputs are acceptable ONLY for advisory (non-decision) use cases

### 4.3 Human Oversight

- High-stakes decisions (financial, medical, safety-critical) require human review
- AI-assisted decisions must clearly indicate AI involvement in all user-facing outputs
- Override mechanisms must exist for all automated AI decisions
