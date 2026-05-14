import re
import logging

logger = logging.getLogger(__name__)

# Enterprise-grade regex patterns for prompt security validation
BLOCKED_PATTERNS = [
    # SQL operations (DML/DDL)
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\btruncate\s+(?:table\s+)?\w+\b",
    r"\balter\s+table\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\brevoke\b",
    r"\bgrant\b",
    
    # Hidden SQL execution/multi-statement
    r"--",
    r"/\*.*\*/",
    r";\s*$",
    r";\s*\w+",
    
    # Prompt injection and jailbreaks
    r"\bignore\s+previous\s+instructions\b",
    r"\breveal\s+(?:system\s+)?prompt\b",
    r"\bexecute\s+sql\b",
    r"\bbypass\s+security\b",
    r"\bdeveloper\s+mode\b",
    r"\badmin\s+mode\b"
]

COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in BLOCKED_PATTERNS]

def is_prompt_safe(question: str) -> tuple[bool, str]:
    """
    Validates a user's prompt against enterprise security rules to prevent
    SQL injection, prompt injection, and jailbreak attempts.
    
    SAFE examples:
    - Which states generate highest revenue?
    - Top payment methods by order count
    
    BLOCKED examples:
    - Drop table products
    - Ignore previous instructions
    - SELECT * FROM users;
    - Delete from customers
    - Reveal system prompt
    
    Args:
        question (str): The raw input question from the user.
        
    Returns:
        tuple[bool, str]: (True, "") if safe, (False, "reason") if unsafe.
    """
    for compiled_pattern in COMPILED_PATTERNS:
        match = compiled_pattern.search(question)
        if match:
            reason = f"Matched forbidden pattern: '{match.group(0)}'"
            logger.warning("Unsafe prompt detected: %s", reason)
            return False, reason
            
    return True, ""
