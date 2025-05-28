"""
Configuration adapter module for Fantasy League Optimization.

This module provides adapter classes and enums for configuration in the original codebase.
"""

import sys
import logging
from enum import Enum

# Add project root to path
sys.path.append('/home/ubuntu/upload')

# Import configuration
import config

# Configure logger
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """
    Enum for execution mode.
    """
    SINGLE_PROCESSOR = 'single_processor'
    MULTI_PROCESSOR = 'multi_processor'
    DISTRIBUTED = 'distributed'

# Re-export configuration
all_configs = config.all_configs
