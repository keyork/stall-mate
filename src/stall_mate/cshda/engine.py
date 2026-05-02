# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from stall_mate.cshda.layer1_extraction.extractor import UDSExtractor
from stall_mate.cshda.layer1_extraction.normalizer import UDSNormalizer
from stall_mate.cshda.layer2_quantification.embedder import Embedder
from stall_mate.cshda.layer2_quantification.formulator import Formulator
from stall_mate.cshda.layer2_quantification.polarity_scorer import PolarityScorer
from stall_mate.cshda.layer3_solver.dispatcher import dispatch
from stall_mate.cshda.layer4_consistency.axiom_checker import AxiomChecker
from stall_mate.cshda.layer4_consistency.audit_logger import AuditLogger
from stall_mate.cshda.layer4_consistency.preference_graph import PreferenceGraph
from stall_mate.cshda.schema.output import (
    AuditTrail,
    ConfidenceBreakdown,
    ConsistencyReport,
    FinalOutput,
)
from stall_mate.cshda.schema.uds import UniversalDecisionSpec

_log = logging.getLogger(__name__)


class CSHDAEngine:
    def __init__(
        self,
        model: str = "glm-5.1",
        base_url: str = "http://localhost:3000/v1",
        api_key: str = "",
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cpu",
        extraction_rounds: int = 3,
        audit_path: Path | None = None,
    ):
        self.extractor = UDSExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            extraction_rounds=extraction_rounds,
        )
        self.normalizer = UDSNormalizer()
        self.embedder = Embedder(model_name=embedding_model, device=device)
        self.scorer = PolarityScorer(self.embedder)
        self.formulator = Formulator(self.embedder, self.scorer)
        self.axiom_checker = AxiomChecker()
        self.preference_graph = PreferenceGraph()
        self.audit_logger = AuditLogger(audit_path)

    def decide(self, natural_input: str) -> FinalOutput:
        _log.info("=== CSHDA Decision Pipeline ===")

        # Layer 1: extraction
        _log.info("Layer 1: Extracting UDS from natural language...")
        uds_raw = self.extractor.extract(natural_input)
        uds = self.normalizer.normalize(uds_raw)
        uds = self.extractor.generate_anchors(uds)
        _log.info("Layer 1 done: %d entities, %d constraints", len(uds.entities), len(uds.constraints))

        # Layer 2: quantification
        _log.info("Layer 2: Quantifying into mathematical formulation...")
        mf = self.formulator.formulate(uds)
        _log.info("Layer 2 done: type=%s confidence=%.2f", mf.decision_type, mf.type_confidence)

        # Layer 3: solving
        _log.info("Layer 3: Solving with %s...", mf.decision_type)
        dr = dispatch(mf)
        _log.info("Layer 3 done: chosen=%s margin=%.4f", dr.chosen, dr.margin)

        # Layer 4: consistency
        _log.info("Layer 4: Checking consistency...")
        consistency = self.axiom_checker.check(dr, mf)
        for eid in dr.chosen:
            pass
        _log.info("Layer 4 done: %s", consistency.model_dump())

        # Build output
        trail = self.audit_logger.build_trail(natural_input, uds, mf, dr, consistency)
        confidence = self._compute_confidence(uds, mf, dr, consistency)

        output = FinalOutput(
            decision=dr,
            consistency_report=consistency,
            confidence_score=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                extraction_stability=uds.metadata.extraction_stability,
                quantification_robustness=mf.type_confidence,
                solution_margin=dr.margin,
            ),
            audit_trail=trail,
        )

        self.audit_logger.log(output)
        _log.info("=== Decision complete: %s (confidence=%.2f) ===", dr.chosen, confidence)
        return output

    def _compute_confidence(
        self,
        uds: UniversalDecisionSpec,
        mf,
        dr,
        consistency: ConsistencyReport,
    ) -> float:
        scores = [
            uds.metadata.extraction_stability,
            mf.type_confidence,
            min(dr.margin * 10, 1.0),
        ]
        if consistency.determinism == "PASS":
            scores.append(1.0)
        if consistency.constraint_satisfaction == "PASS":
            scores.append(1.0)
        return sum(scores) / len(scores) if scores else 0.5
