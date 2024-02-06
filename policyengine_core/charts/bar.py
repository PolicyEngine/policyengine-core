import pandas as pd
from .formatting import *
import plotly.express as px
from microdf import MicroSeries
from typing import Callable
import numpy as np


def bar_chart(
    data: pd.Series,
    showlegend: bool = False,
    remove_zero_index: bool = True,
    text_format: str = "+.1%",
    positive_colour: str = BLUE,
    negative_colour: str = DARK_GRAY,
    hover_text_function: Callable = None,
    **kwargs,
):
    """Create a PolicyEngine bar chart.

    Args:
        data (pd.Series): A pandas series.
        showlegend (bool, optional): Whether to show the legend. Defaults to False.
        remove_zero_index (bool, optional): Whether to remove the zero index. Defaults to True.
        text_format (str, optional): The format of the text. Defaults to "+.1%".
        positive_colour (str, optional): The colour of positive values. Defaults to BLUE.
        negative_colour (str, optional): The colour of negative values. Defaults to DARK_GRAY.
        hover_text_labels (list, optional): The hover text labels. Defaults to None.

    Returns:
        go.Figure: A plotly figure.
    """

    hover_text_labels = [
        (
            hover_text_function(index, value)
            if hover_text_function is not None
            else None
        )
        for index, value in data.items()
    ]

    fig = (
        px.bar(
            data,
            text=data.apply(lambda x: f"{x:{text_format}}"),
            custom_data=[hover_text_labels] if hover_text_labels else None,
        )
        .update_layout(
            showlegend=showlegend,
            uniformtext=dict(
                mode="hide",
                minsize=12,
            ),
            **kwargs,
        )
        .update_traces(
            marker_color=[
                positive_colour if v > 0 else negative_colour
                for v in data.values
            ],
            hovertemplate=(
                "%{customdata[0]}<extra></extra>"
                if hover_text_labels is not None
                else None
            ),
        )
    )
    return format_fig(fig)


def cross_section_bar_chart(
    data: MicroSeries,
    cross_section: MicroSeries,
    slices: list = [-0.05, -0.01, 0.01, 0.05],
    add_infinities: bool = True,
    text_format: str = ".1%",
    hover_text_function: Callable = None,
    category_names=None,
    color_discrete_map: dict = None,
    **kwargs,
):
    df = pd.DataFrame()
    slices = [-np.inf, *slices, np.inf] if add_infinities else slices
    for i, lower, upper in zip(range(len(slices)), slices[:-1], slices[1:]):
        for cross_section_value in cross_section.unique():
            in_slice = (data >= lower) * (data < upper)
            value = (
                data[cross_section == cross_section_value][in_slice].count()
                / data[cross_section == cross_section_value].count()
            )
            category = (
                category_names[i]
                if category_names is not None
                else f"{lower:.0%} to {upper:.0%}"
            )
            row = {
                "Category": category,
                "Cross section": cross_section_value,
                "Value": value,
                "Hover text": (
                    hover_text_function(cross_section_value, category, value)
                    if hover_text_function is not None
                    else None
                ),
            }
            df = df.append(row, ignore_index=True)

    fig = (
        px.bar(
            df,
            x="Value",
            y="Cross section",
            color="Category",
            barmode="stack",
            orientation="h",
            text=df["Value"].apply(lambda x: f"{x:{text_format}}"),
            color_discrete_map=color_discrete_map,
            custom_data=["Hover text"],
        )
        .update_layout(
            xaxis=dict(
                tickformat="%",
                title="Fraction of observations",
            ),
            yaxis=dict(tickvals=list(range(1, 11))),
            uniformtext=dict(
                mode="hide",
                minsize=12,
            ),
            **kwargs,
        )
        .update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
    )
    return format_fig(fig)
