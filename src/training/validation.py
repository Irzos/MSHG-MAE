"""
Unified validation controller.

Provides a clear, consistent interface to control validation frequency and
scope, consolidating previously scattered parameters and logic.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ValidationController:
    """Centralized controller for validation scheduling."""
    
    def __init__(self, validation_config: Dict[str, Any], max_steps: int):
        """
        Initialize controller.

        Args:
            validation_config: Validation config dict
            max_steps: Max training steps (for default interval)
        """
        self.enabled = validation_config.get('enabled', True)
        self.interval_steps = validation_config.get('interval_steps', max(1, max_steps // 8))
        self.quick_batches = validation_config.get('quick_batches', 50)
        
        # Tuning mode: very large interval effectively disables validation
        self.is_tuning_mode = self.interval_steps > max_steps
        
        if self.is_tuning_mode:
            logger.info(f"Validation Controller: tuning mode, validation disabled (interval={self.interval_steps})")
        else:
            logger.info(f"Validation Controller: normal mode, validate every {self.interval_steps} steps")
    
    def should_validate(self, current_step: int, step_updated: bool = True) -> bool:
        """
        Decide whether to run validation at this step.

        Args:
            current_step: Current training step
            step_updated: Whether optimizer step actually occurred (skip on accumulation)

        Returns:
            bool: True if validation should run
        """
        if not self.enabled or not step_updated:
            return False
            
        if self.is_tuning_mode:
            return False
            
        return current_step % self.interval_steps == 0
    
    def get_quick_batches(self) -> Optional[int]:
        """Optional limit for quick validation batches."""
        return self.quick_batches if self.quick_batches > 0 else None
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get controller status for logging."""
        return {
            'enabled': self.enabled,
            'interval_steps': self.interval_steps,
            'quick_batches': self.quick_batches,
            'is_tuning_mode': self.is_tuning_mode
        }
