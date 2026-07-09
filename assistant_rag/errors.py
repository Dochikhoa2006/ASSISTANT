"""Repository and pipeline exception types."""

class RepositoryConflictError(Exception):
    """Base class for database concurrency/conflict errors."""
    pass

class KnowledgeConflictError(RepositoryConflictError):
    """Raised when knowledge mutation fails due to a version or status mismatch."""
    pass

class ReminderConflictError(RepositoryConflictError):
    """Raised when reminder mutation fails due to a version or status mismatch."""
    pass

class RepositoryValidationError(Exception):
    """Raised when an action target is invalid or no longer meets requirements."""
    pass

class RepositoryTransactionError(Exception):
    """Raised for unexpected database errors that result in a transaction rollback."""
    pass


def safe_repository_reason(exc: Exception) -> str:
    """Map raw exception types to safe, user-facing summary strings."""
    if isinstance(exc, KnowledgeConflictError):
        return "The knowledge item changed before the update could be completed. Please try again."
    if isinstance(exc, ReminderConflictError):
        return "The reminder changed before the update could be completed. Please try again."
    if isinstance(exc, RepositoryValidationError):
        return "The action could not be completed because the target is no longer valid."
    if isinstance(exc, RepositoryTransactionError):
        return "The database transaction failed and was rolled back."
    
    return "An unexpected error occurred during the transaction."
