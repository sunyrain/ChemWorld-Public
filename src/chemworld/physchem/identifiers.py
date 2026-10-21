"""Pure canonical identifier checks shared by component inputs and data readers."""

import re
from numbers import Integral

_CAS = re.compile(r"[1-9][0-9]{1,6}-[0-9]{2}-[0-9]")


def validate_cas_number(value):
    if not isinstance(value, str) or _CAS.fullmatch(value) is None:
        raise ValueError("cas_number must use CAS Registry Number format (2-7/2/1 ASCII digits)")
    digits = value.replace("-", "")
    checksum = sum(int(digit) * weight for weight, digit in enumerate(digits[-2::-1], 1)) % 10
    if checksum != int(digits[-1]):
        raise ValueError("cas_number checksum is invalid")
    return value


def check_cas_number(value):
    try:
        validate_cas_number(value)
    except ValueError:
        return False
    return True


def cas_to_int(value):
    return int(validate_cas_number(value).replace("-", ""))


def int_to_cas(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("CAS integer representation must be a nonnegative integer")
    digits = str(value)
    if not 5 <= len(digits) <= 10:
        raise ValueError("CAS integer has an invalid number of digits")
    return validate_cas_number(f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}")


def sorted_cas_key(values):
    return tuple(sorted(values, key=cas_to_int))
