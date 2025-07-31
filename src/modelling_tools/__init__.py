from .seasonality import (
    generate_daily_seasonality,
    generate_weekly_seasonality,
    generate_yearly_seasonality,
)

def main() -> None:
    # works when you run `uv run modelling_tools`
    # because `modelling-tools = "modelling_tools:main"` in pyproject.toml
    print("Hello from modelling-tools!")
    print(
        (
            generate_daily_seasonality,
            generate_weekly_seasonality,
            generate_yearly_seasonality,
        )
    )
