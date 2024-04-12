import plotly.graph_objects as go
from IPython.display import HTML


GREEN = "#29d40f"
LIGHT_GREEN = "#C5E1A5"
DARK_GREEN = "#558B2F"
BLUE_LIGHT = "#D8E6F3"
BLUE_PRIMARY = BLUE = "#2C6496"
BLUE_PRESSED = "#17354F"
BLUE_98 = "#F7FAFD"
TEAL_LIGHT = "#D7F4F2"
TEAL_ACCENT = "#39C6C0"
TEAL_PRESSED = "#227773"
DARKEST_BLUE = "#0C1A27"
DARK_GRAY = "#616161"
GRAY = "#808080"
LIGHT_GRAY = "#F2F2F2"
MEDIUM_DARK_GRAY = "#D2D2D2"
WHITE = "#FFFFFF"
TEAL_98 = "#F7FDFC"
BLACK = "#000000"

BLUE_COLOUR_SCALE = [
    BLUE_LIGHT,
    BLUE_PRIMARY,
    BLUE_PRESSED,
]


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
            x=1.1,
            y=-0.15,
            sizex=0.15,
            sizey=0.15,
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
