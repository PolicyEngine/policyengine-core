import requests
import time
import plotly.graph_objects as go
from .formatting import (
    DARK_GRAY,
    MEDIUM_DARK_GRAY,
    WHITE,
    BLUE_LIGHT,
    BLUE_PRIMARY,
    format_fig,
)


def get_api_chart_data(
    country_id: str,
    reform_policy_id: int,
    chart_key: str,
    region: str,
    time_period: str,
    baseline_policy_id: int = None,
    version: str = None,
) -> dict:
    if baseline_policy_id is None or version is None:
        response = requests.get(
            f"https://api.policyengine.org/{country_id}/metadata"
        )
        result = response.json().get("result", {})
        baseline_policy_id = result.get("current_law_id")
        version = result.get("version")
    url = f"https://api.policyengine.org/{country_id}/economy/{reform_policy_id}/over/{baseline_policy_id}?region={region}&time_period={time_period}&version={version}"
    response = requests.get(url)
    if not response.ok:
        raise ValueError(response.text)
    json = response.json()
    while json.get("status") == "computing":
        time.sleep(1)
        response = requests.get(url)
        if not response.ok:
            raise ValueError(response.text)
        json = response.json()
    return json.get("result", {}).get(chart_key)


# Specific chart definitions


def intra_decile_chart(
    country_id: str,
    reform_policy_id: int,
    region: str,
    time_period: str,
    baseline_policy_id: int = None,
) -> go.Figure:
    impact = get_api_chart_data(
        country_id=country_id,
        reform_policy_id=reform_policy_id,
        chart_key="intra_decile",
        baseline_policy_id=baseline_policy_id,
        region=region,
        time_period=time_period,
    )
    impact = {"intra_decile": impact}

    decile_numbers = list(range(1, 11))
    outcome_labels = [
        "Gain more than 5%",
        "Gain less than 5%",
        "No change",
        "Lose less than 5%",
        "Lose more than 5%",
    ]
    outcome_colours = [
        DARK_GRAY,
        MEDIUM_DARK_GRAY,
        WHITE,
        BLUE_LIGHT,
        BLUE_PRIMARY,
    ][::-1]

    data = []

    for outcome_label, outcome_colour in zip(outcome_labels, outcome_colours):
        data.append(
            {
                "type": "bar",
                "y": ["All"],
                "x": [impact["intra_decile"]["all"][outcome_label]],
                "name": outcome_label,
                "legendgroup": outcome_label,
                "offsetgroup": outcome_label,
                "marker": {
                    "color": outcome_colour,
                },
                "orientation": "h",
                "text": [
                    f"{impact['intra_decile']['all'][outcome_label] * 100:.0f}%"
                ],
                "textposition": "inside",
                "textangle": 0,
                "xaxis": "x",
                "yaxis": "y",
                "showlegend": False,
                "hoverinfo": "none",
            }
        )

    for outcome_label, outcome_colour in zip(outcome_labels, outcome_colours):
        data.append(
            {
                "type": "bar",
                "y": decile_numbers,
                "x": impact["intra_decile"]["deciles"][outcome_label],
                "name": outcome_label,
                "marker": {
                    "color": outcome_colour,
                },
                "orientation": "h",
                "text": [
                    f"{value * 100:.0f}%"
                    for value in impact["intra_decile"]["deciles"][
                        outcome_label
                    ]
                ],
                "textposition": "inside",
                "textangle": 0,
                "xaxis": "x2",
                "yaxis": "y2",
                "hoverinfo": "none",
            }
        )

    layout = {
        "barmode": "stack",
        "grid": {
            "rows": 2,
            "columns": 1,
        },
        "yaxis": {
            "title": "",
            "tickvals": ["All"],
            "domain": [0.91, 1],
        },
        "xaxis": {
            "title": "",
            "tickformat": ".0%",
            "anchor": "y",
            "matches": "x2",
            "showgrid": False,
            "showticklabels": False,
        },
        "xaxis2": {
            "title": "Population share",
            "tickformat": ".0%",
            "anchor": "y2",
        },
        "yaxis2": {
            "title": "Income decile",
            "tickvals": decile_numbers,
            "anchor": "x2",
            "domain": [0, 0.85],
        },
        "uniformtext": {
            "mode": "hide",
            "minsize": 10,
        },
    }

    return format_fig(go.Figure(data=data, layout=layout)).update_traces(
        marker_line_width=0,
        width=[0.9] * 10,
    )


def decile_chart(
    country_id: str,
    reform_policy_id: int,
    region: str,
    time_period: str,
    baseline_policy_id: int = None,
) -> go.Figure:
    impact = get_api_chart_data(
        country_id=country_id,
        reform_policy_id=reform_policy_id,
        chart_key="decile",
        baseline_policy_id=baseline_policy_id,
        region=region,
        time_period=time_period,
    )
    impact = {"decile": impact}

    decile_numbers = list(range(1, 11))
    # Sort deciles by key order 1 to 10
    decile_values = []
    for i in decile_numbers:
        decile_values.append(impact["decile"]["relative"][str(i)])

    data = [
        {
            "x": decile_numbers,
            "y": decile_values,
            "type": "bar",
            "marker": {
                "color": [
                    DARK_GRAY if value < 0 else BLUE_PRIMARY
                    for value in decile_values
                ],
            },
            "text": [f"{value:+.1%}" for value in decile_values],
            "textangle": 0,
        }
    ]

    layout = {
        "xaxis": {
            "title": "Income decile",
            "tickvals": decile_numbers,
        },
        "yaxis": {
            "title": "Relative change",
            "tickformat": "+,.0%",
        },
        "uniformtext": {
            "mode": "hide",
            "minsize": 8,
        },
    }

    return format_fig(go.Figure(data=data, layout=layout))
