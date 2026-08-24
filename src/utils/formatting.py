"""Small display-formatting helpers used by the UI layer."""


def format_days(value: float) -> str:
    days = round(value)
    unit = "day" if days == 1 else "days"
    return f"{days} {unit}"


def format_signed_days(value: float) -> str:
    days = round(value)
    sign = "+" if days >= 0 else ""
    unit = "day" if abs(days) == 1 else "days"
    return f"{sign}{days} {unit}"


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"
