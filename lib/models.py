import torch.nn as nn
import torch


class LinearModel(nn.Module):
    """
    Simple linear NN.
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class SingleParameterFormula(nn.Module):
    """
    Single-variable function.
    """
    def __init__(self, depth=0):
        super(SingleParameterFormula, self).__init__()
        self._depth = depth
        self._subformula = None
        if self._depth == 0:
            # When depth is zero, formula is just a real number
            new_lambda = nn.Parameter((2 * torch.randn((1, 1))), requires_grad=True)
            self.register_parameter('lambda_0', new_lambda)
            return
        self._inner_subformula = None
        if self._depth > 1:
            self._inner_subformula = SingleParameterFormula(self._depth - 1)
        new_lambda = nn.Parameter((2 * torch.randn((1, 1))), requires_grad=True)
        new_power = nn.Parameter((2 * torch.randn((1, 1))), requires_grad=True)
        self.register_parameter('lambda_1', new_lambda)
        self.register_parameter('power_1', new_power)
        self._outer_subformula = SingleParameterFormula(self._depth - 1)

    def forward(self, x):
        """
        Iterate over subformulas, recursively computing result using results of subformulas
        """
        # When depth is 0, we just return the corresponding number
        if self._depth == 0:
            lambda0 = self.__getattr__('lambda_0')
            return lambda0.repeat(x.shape[0], 1).to(x.device)

        ans = torch.zeros(x.shape[0], 1).to(x.device)
        lambda1 = self.__getattr__('lambda_1')
        power1 = self.__getattr__('power_1')
        x_powered = torch.t(torch.pow(x[:, 0], power1))
        subformula_result = torch.ones((x.shape[0], 1)).to(x.device)
        if self._depth > 1:
            subformula_result = self._inner_subformula(x)
        ans += lambda1 * x_powered * subformula_result
        ans += self._outer_subformula(x)
        return ans

    def __repr__(self):
        """
        Return tex-style string, recursively combining result from representation of subformulas
        """
        if self._depth == 0:
            return '{:.3}'.format(self.__getattr__('lambda_0').item())

        lambda1 = self.__getattr__('lambda_1').item()
        power1 = self.__getattr__('power_1').item()
        if self._depth > 1:
            ans = '({:.3}x^{{{:.3}}} {} + {})'.format(
                lambda1, power1, str(self._inner_subformula), str(self._outer_subformula))
        else:
            ans = '({:.3}x^{{{:.3}}} + {})'.format(lambda1, power1, str(self._outer_subformula))
        return ans
