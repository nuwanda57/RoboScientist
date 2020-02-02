class UnimplementedError(Exception):
    """Raised when no Implementation exists."""
    pass


class BaseSuccessCriterion(object):
    def is_satisfied(self, **kwargs):
        raise UnimplementedError


class LossSuccessCriterion(BaseSuccessCriterion):
    def __init__(self, threshold=0.1):
        self._threshold = threshold

    def is_satisfied(self, **kwargs):
        return kwargs['test_mse'] is not None and kwargs['test_mse'] < self._threshold
