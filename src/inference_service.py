import time
import datetime
import uuid
from typing import Dict, Any

from src.data_validator import DataValidator
from src.recovery_engine import RecoveryEngine
from src.decision_trace import DecisionTracer
from src.audit_trail import AuditTrail

class RecoveryInferenceService:
    """
    Real-time/Event-driven inference service layer.
    Accepts ONE incoming failed-payment event, validates it, runs the ML prediction,
    assigns a recovery decision, generates a trace, and produces an audit record.
    """

    def __init__(self, simulator_cost: float = 50.0, simulator_threshold: float = 0.05):
        # Initialize internal modules
        self.validator = DataValidator()
        self.engine = RecoveryEngine()
        self.tracer = DecisionTracer(simulator_cost=simulator_cost, simulator_threshold=simulator_threshold)
        self.auditor = AuditTrail(simulator_cost=simulator_cost, simulator_threshold=simulator_threshold)

    def predict_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single failed payment event through the recovery decision pipeline.
        """
        start_time = time.perf_counter()
        
        # 1. Identity processing
        payment_id = str(event.get('payment_id', str(uuid.uuid4())))
        customer_id = event.get('customer_id')

        result = {
            'event_identity': {
                'payment_id': payment_id,
                'customer_id': customer_id
            },
            'validation': {},
            'prediction': {},
            'decision': {},
            'economic_estimate': {},
            'explanation': {},
            'processing_metadata': {}
        }

        # 2. Validation
        val_result = self.validator.validate_record(event)
        result['validation'] = {
            'is_valid': val_result.is_valid,
            'errors': val_result.errors
        }
        
        if not val_result.is_valid:
            result['processing_metadata'] = {
                'status': 'error',
                'error_type': 'validation_error',
                'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2)
            }
            return result

        try:
            # 3. ML Prediction and Decision (via RecoveryEngine)
            engine_result = self.engine.predict_recovery(event)
            
            # Validate output prob
            prob = engine_result.get('recovery_probability')
            prob_val = self.validator.validate_prediction_probability(prob)
            if not prob_val.is_valid:
                result['validation']['is_valid'] = False
                result['validation']['errors'].extend(prob_val.errors)
                result['processing_metadata'] = {
                    'status': 'error',
                    'error_type': 'model_output_error',
                    'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2)
                }
                return result

            # 4. Decision Trace Generation
            trace = self.tracer.generate_trace(event, engine_result)

            # 5. Populate Result structure safely
            result['prediction'] = {
                'recovery_probability': prob
            }
            
            result['decision'] = {
                'recommended_action': engine_result.get('recommended_action', 'Unknown'),
                'recovery_priority': engine_result.get('priority', 'UNKNOWN')
            }
            
            result['economic_estimate'] = {
                'expected_recovery': engine_result.get('expected_recovery', 0.0),
                'effective_retry_cost': trace.get('effective_retry_cost', 0.0),
                'expected_retry_net_value': trace.get('expected_retry_net_value', 0.0)
            }
            
            result['explanation'] = {
                'decision_reason': trace.get('decision_reason', ''),
                'key_input_factors': trace.get('key_input_factors', [])
            }

            # 6. Audit Compatibility (generate an audit record representation)
            audit_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # inject payment_id into event if it was missing but we generated one for traceability
            event_for_audit = dict(event)
            event_for_audit['payment_id'] = payment_id
            
            audit_record = self.auditor.create_audit_record(
                record=event_for_audit,
                engine_result=engine_result,
                strategy_name="Rule-Based Real-Time Default",
                audit_timestamp=audit_ts
            )
            result['audit_record'] = audit_record
            
            result['processing_metadata'] = {
                'status': 'success',
                'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2),
                'timestamp_utc': audit_ts
            }
            
        except Exception as e:
            result['processing_metadata'] = {
                'status': 'error',
                'error_type': 'pipeline_exception',
                'error_message': 'Inference pipeline failed. Please retry or inspect server logs.',
                'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2)
            }

        return result
