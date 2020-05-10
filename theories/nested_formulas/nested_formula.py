import torch.nn as nn

from theories.nested_formulas.auxiliary_functions import *


class NestedFormula(nn.Module):
    """
    Class used for representing formulas
    
    Attributes:
        depth
        num_variables
        subformulas - list of subformulas of smaller depth, which are used for computing
    """

    def __init__(self, depth=0, num_variables=1,
                 #                  functions=["tan", "sin", "cos", "ln", "atan", "asin", "acos"]
                 ):
        super(NestedFormula, self).__init__()
        self.depth = depth
        self.num_variables = num_variables
        self.subformulas = nn.ModuleList()
        #         self.map_names_to_functions = {
        #             "tan": torch.tan,
        #             "sin": torch.sin,
        #             "cos": torch.cos,
        #             "atan": torch.atan,
        #             "asin": torch.asin,
        #             "acos": torch.acos,
        #             "ln": torch.log
        #         }
        #         self.functions = functions

        # When depth is zero, formula is just a real number
        if depth == 0:
            new_lambda = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
            self.register_parameter("lambda_0", new_lambda)
            new_rational_lambda = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
            self.register_parameter("rational_lambda_0", new_rational_lambda)
        else:
            for i in range(self.num_variables):
                # When depth is 1, we do not need to create subformulas, since they would be just real numbers
                if self.depth != 1:
                    subformula = NestedFormula(self.depth - 1, self.num_variables)
                    self.subformulas.append(subformula)
                new_lambda = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
                new_power = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
                new_rational_lambda = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
                new_rational_power = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
                self.register_parameter("lambda_{}".format(i), new_lambda)
                self.register_parameter("power_{}".format(i), new_power)
                self.register_parameter("rational_lambda_{}".format(i), new_rational_lambda)
                self.register_parameter("rational_power_{}".format(i), new_rational_power)

            #                 for function in functions:
            #                     new_lambda = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
            #                     self.register_parameter("lambda_{}".format(function), new_lambda)

            self.last_subformula = NestedFormula(self.depth - 1, self.num_variables)

    def forward(self, x):
        """
        Iterate over subformulas, recursively computing result using results of subformulas
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # When depth is 0, we just return the corresponding number
        if self.depth == 0:
            return self.get_lambda(0).repeat(x.shape[0], 1).to(x.device)

        ans = torch.zeros(x.shape[0], 1).to(x.device)
        for i in range(self.num_variables):
            x_powered = torch.t(x[:, i] ** self.get_power(i))
            subformula_result = torch.ones((x.shape[0], 1)).to(x.device)
            # When depth is 1, we do not need to compute subformulas
            if self.depth != 1:
                subformula_result = self.subformulas[i](x)
            ans += self.get_lambda(i) * x_powered * subformula_result

            # Here I should modify my code in order to make it possible to use sine cosine etc.
            # But this is not urgent, so I probably won't do that.

        ans += self.last_subformula(x)
        return ans

    def cuda(self, device=None):
        self = super().cuda(device)
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self

    def simplify(self, X_val, y_val, max_denominator=10, inplace=False):
        """
        Simplifies the formula, iterating over all its parameters and trying to substitute them with close rational number
        This function was not properly tested and, generally, is not supposed to work yet.
        Parameters:
            X_val: torch.tensor, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y_val: torch.tensor, shape (n_samples, 1)
                Target vector relative to X.
            max_denominator: int
                algorithm tries rational numbers with denominator not greater than max_denominator
            inplace: bool
                if True, when modify the original formula
                otherwise return new formula, leaving original one unchanged
        Returns:
            self, if inplace set to True
            simplified_version otherwise
        """

        simplified_version = copy.deepcopy(self)
        simplified_state_dict = simplified_version.state_dict()

        # Iterate over all parameters
        for key, value in self.state_dict().items():
            if "rational" not in key:  # We do not simplify rational parameters - they will be the result of simplification
                simplified_version_for_iteration = copy.deepcopy(simplified_version)
                simplified_state_dict_for_iteration = simplified_version_for_iteration.state_dict()
                y_predict = simplified_version(X_val)
                loss = nn.MSELoss()(y_val, y_predict)
                descriptive_length_of_loss = descriptive_length_of_real_number(loss)
                descriptive_length_of_existing_parameter = descriptive_length_of_real_number(value)

                # Iterate over all possible denominators
                for possible_denominator in range(1, max_denominator + 1):
                    #                     print("trying denominator", possible_denominator)
                    simplified_parameter_numerator = torch.round(value * possible_denominator)
                    simplified_state_dict_for_iteration[key] = simplified_parameter_numerator / possible_denominator
                    simplified_version_for_iteration.load_state_dict(simplified_state_dict_for_iteration)
                    descriptive_length_of_simplified_parameter = descriptive_length_of_fraction(
                        simplified_parameter_numerator, possible_denominator)
                    #                     print(simplified_parameter_numerator, possible_denominator)
                    y_predict_simplified = simplified_version_for_iteration(X_val)
                    loss_of_simplified_model = nn.MSELoss()(y_val, y_predict_simplified)
                    descriptive_length_of_loss_of_simplified_model = descriptive_length_of_real_number(
                        loss_of_simplified_model)
                    # If the descriptive length did not improve, revert the change.
                    #                     print("descriptive_length_of_loss_of_simplified_model", descriptive_length_of_loss_of_simplified_model)
                    #                     print("descriptive_length_of_simplified_parameter", descriptive_length_of_simplified_parameter)
                    #                     print("descriptive_length_of_loss", descriptive_length_of_loss)
                    #                     print("descriptive_length_of_existing_parameter", descriptive_length_of_existing_parameter)

                    if descriptive_length_of_loss_of_simplified_model + descriptive_length_of_simplified_parameter > descriptive_length_of_loss + descriptive_length_of_existing_parameter:
                        simplified_version_for_iteration.load_state_dict(simplified_state_dict)
                    else:
                        # If we are successful, we update everything
                        simplified_state_dict[add_rational_in_name(key)] = torch.tensor(
                            [simplified_parameter_numerator, possible_denominator])
                        simplified_version.load_state_dict(simplified_state_dict)
                        simplified_version_for_iteration = copy.deepcopy(simplified_version)
                        simplified_state_dict_for_iteration = simplified_version_for_iteration.state_dict()

                simplified_state_dict = simplified_state_dict_for_iteration
                simplified_version.load_state_dict(simplified_state_dict)

        if inplace:
            self = copy.deepcopy(simplified_version)
        else:
            return simplified_version

    def get_lambda(self, i):
        return self.__getattr__('lambda_{}'.format(i))

    def get_rational_lambda(self, i):
        return self.__getattr__('rational_lambda_{}'.format(i))

    def get_power(self, i):
        return self.__getattr__('power_{}'.format(i))

    def get_rational_power(self, i):
        return self.__getattr__('rational_power_{}'.format(i))

    def __repr__(self):
        """
        Return tex-style string, recursively combining result from representation of subformulas
        """
        if self.depth == 0:
            if self.get_rational_lambda(0)[1] > 0:  # if it is equal to 0, it means that there is no rational value
                return form_fraction_representation(self.get_rational_lambda(0))
            return form_real(self.get_lambda(0))

        ans = ["\left("]
        for i in range(self.num_variables):
            # First we add lambda
            if i != 0 and self.get_lambda(i) > 0:
                ans.append(" + ")
            if self.get_rational_lambda(i)[1] > 0:
                ans.append(form_fraction_representation(self.get_rational_lambda(i)))
            else:
                ans.append(form_real(self.get_lambda(i)))
                # Then we add variable and its power
            ans.append("x_{}^".format(i + 1) + "{")
            if self.get_rational_power(i)[1] > 0:
                ans.append(form_fraction_representation(self.get_rational_power(i)))
            else:
                ans.append(form_real(self.get_power(i)))
            ans += "}"
            # Then we add the corresponding subformula
            if self.depth != 1:
                ans.append(str(self.subformulas[i]))
        if self.last_subformula.get_lambda(0) > 0:
            ans.append(" + ")
        ans.append(str(self.last_subformula))
        ans.append(r"\right)")
        ans = ''.join(ans)
        return ans
