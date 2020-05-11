import torch
from IPython.core.display import display, Math


def descriptive_length_of_fraction(numerator, denominator):
    return torch.log2((1 + torch.abs(torch.tensor(denominator))) * torch.abs(torch.tensor(numerator)))


def logplus(number):
    return torch.log2(1 + number ** 2) / 2


def descriptive_length_of_real_number(real_number, precision_floor=1e-8):
    return logplus(real_number / torch.tensor(precision_floor))


def form_fraction_representation(fraction: torch.tensor) -> str:
    if fraction[1].item() != 1:
        return r"\frac{" + str(int(fraction[0].item())) + "}{" + str(int(fraction[1].item())) + "}"
    return str(int(fraction[0].item()))


def form_real(number: torch.tensor) -> str:
    return "{:.3f}".format(number.item())


def add_rational_in_name(name: str) -> str:
    if 'lambda' in name:
        position = name.find('lambda')
    else:
        position = name.find('power')
    return name[:position] + "rational_" + name[position:]