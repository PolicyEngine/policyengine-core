class ParameterPathError(ValueError):
    """
    Exception raised when there's an error in parameter path resolution.
    Note: This class exists for backwards compatibility but is not currently used.
    The get_parameter function uses ValueError instead.
    """
    
    def __init__(self, message, parameter_path=None, failed_at=None):
        """
        Args:
            message (str): The error message.
            parameter_path (str, optional): The full parameter path that was being accessed.
            failed_at (str, optional): The specific component in the path where the failure occurred.
        """
        self.parameter_path = parameter_path
        self.failed_at = failed_at
        super(ParameterPathError, self).__init__(message)