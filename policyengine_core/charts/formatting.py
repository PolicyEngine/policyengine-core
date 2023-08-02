import plotly.graph_objects as go
from IPython.display import HTML


WHITE = "#FFF"
BLUE = "#2C6496"
GRAY = "#BDBDBD"
MEDIUM_DARK_GRAY = "#D2D2D2"
DARK_GRAY = "#616161"
GREEN = "#29d40f"
LIGHT_GRAY = "#F2F2F2"
LIGHT_GREEN = "#C5E1A5"
DARK_GREEN = "#558B2F"
BLACK = "#000"

BLUE_COLOR_SCALE = [BLUE, "#265782", "#20496E", "#1A3C5A"]


def format_fig(fig: go.Figure) -> go.Figure:
    """Format a plotly figure to match the PolicyEngine style guide.

    Args:
        fig (go.Figure): A plotly figure.

    Returns:
        go.Figure: A plotly figure with the PolicyEngine style guide applied.
    """
    fig.update_layout(
        font=dict(
            family="Roboto Serif",
            color="black",
        )
    )
    fig.add_layout_image(
        dict(
            source="https://raw.githubusercontent.com/PolicyEngine/policyengine-app/master/src/images/logos/policyengine/blue.png",
            xref="paper",
            yref="paper",
            x=1,
            y=-0.15,
            sizex=0.2,
            sizey=0.2,
            xanchor="right",
            yanchor="bottom",
        )
    )

    # set template
    fig.update_layout(
        template="plotly_white",
        height=600,
        width=800,
    )
    # don't show modebar
    fig.update_layout(
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(0,0,0,0)",
        )
    )
    return fig


def display_fig(fig: go.Figure) -> HTML:
    return HTML(
        format_fig(fig).to_html(full_html=False, include_plotlyjs="cdn")
    )


def cardinal(n: int) -> int:
    """Convert an integer to a cardinal string."""
    ending_number = n % 10
    if ending_number == 1:
        return f"{n}st"
    elif ending_number == 2:
        return f"{n}nd"
    elif ending_number == 3:
        return f"{n}rd"
    else:
        return f"{n}th"
