from .seasonality import (
    add_daily_seasonality,
    add_weekly_seasonality,
    add_yearly_seasonality,
)

def main() -> None:
    # works when you run `uv run modelling_tools`
    # because `modelling-tools = "modelling_tools:main"` in pyproject.toml
    print("Hello from modelling-tools!")
    print(
        (
            add_daily_seasonality,
            add_weekly_seasonality,
            add_yearly_seasonality,
        )
    )
