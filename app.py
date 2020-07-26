import io
from typing import List, Optional
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import markdown

import cornac
import streamlit as st

COLOR = "grey"
BACKGROUND_COLOR = "#fff"
RANDOM_NAME = np.random.choice(["David", "Kai Yuan", "Kai Leng"])
READ_COUNT1 = int(np.random.random(1)*10)
READ_COUNT2 = int(np.random.random(1)*50)
READ_COUNT3 = int(np.random.random(1)*500)
CURR_READ = int(np.random.random(1)*10)


def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {"#696969"};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def select_block_container_style():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.subheader("Block Container Style")
    max_width_100_percent = st.sidebar.checkbox("Max-width: 100%?", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Select max-width in px", 100, 2000, 1200, 100)
    else:
        max_width = 1200
    dark_theme = st.sidebar.checkbox("Dark Theme?", False)
    padding_top = st.sidebar.number_input("Select padding top in rem", 0, 200, 5, 1)
    padding_right = st.sidebar.number_input("Select padding right in rem", 0, 200, 1, 1)
    padding_left = st.sidebar.number_input("Select padding left in rem", 0, 200, 1, 1)
    padding_bottom = st.sidebar.number_input(
        "Select padding bottom in rem", 0, 200, 10, 1
    )
    if dark_theme:
        global COLOR
        global BACKGROUND_COLOR
        BACKGROUND_COLOR = "rgb(17,17,17)"
        COLOR = "#fff"

    _set_block_container_style(
        max_width,
        max_width_100_percent,
        padding_top,
        padding_right,
        padding_left,
        padding_bottom,
    )


class Grid:
    """A (CSS) Grid"""

    def __init__(
        self,
        template_columns="1 1 1",
        gap="10px",
        background_color=COLOR,
        color=BACKGROUND_COLOR,
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    grid-template-columns: {self.template_columns};
    grid-gap: {self.gap};
    background-color: {self.background_color};
    color: {self.color};
    }}
    .box {{
    background-color: {self.color};
    color: {self.background_color};
    border-radius: 5px;
    padding: 20px;
    font-size: 150%;
    }}
    table {{
        color: {self.color}
    }}
</style>
"""

    def _get_cells_style(self):
        return (
            "<style>"
            + "\n".join([cell._to_style() for cell in self.cells])
            + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell._to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell


class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()

    def plotly_chart(self, fig):
        self.inner_html = f"""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<body>
    <p>This should have been a plotly plot.
    But since *script* tags are removed when inserting MarkDown/ HTML i cannot get it to workto work.
    But I could potentially save to svg and insert that.</p>
    <div id='divPlotly'></div>
    <script>
        var plotly_data = {fig.to_json()}
        Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
    </script>
</body>
"""

    def pyplot(self, fig=None, **kwargs):
        string_io = io.StringIO()
        plt.savefig(string_io, format="svg", fig=(2, 2))
        svg = string_io.getvalue()[215:]
        plt.close(fig)
        self.inner_html = '<div height="200px">' + svg + "</div>"

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


def load_object(filename):
    """
    Load a pickled object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_title_info(comics_df, book_ids: list):
    """
    Takes a list of book_ids and converts them to their titles

    Attributes:
        comics_df {pandas DataFrame}: dataframe from books_comics_graphic
        book_ids {list}: list of book_id to get title info

    Returns:
        {list}: corresponding titles
    """
    return [comics_df[comics_df.book_id == str(i)].title.values[0]
            for i in book_ids]


def get_recommendations(model, userid, comics_df, k=10):
    """
    Attributes:
        model {instance of trained cornac model}
        userid {int}: p_id

    Returns:
        {list}: top-k recommended items' title
    """
    user = model.train_set.uid_map[userid]
    recommendations, _ = model.rank(user)

    item_idx2id = list(model.train_set.item_ids)
    top_k_recs = [item_idx2id[i] for i in recommendations[:k]]

    return get_title_info(comics_df, top_k_recs), top_k_recs


def get_historical_txn(model, userid, hist_df, k=5):
    """Get historical transactions of user

    Args:
        model {instance of trained cornac model}
        userid {str}
        hist_df {pandas DataFrame}: from historical_txn.pkl
        k {int, optional}: No of historical txns to show. Defaults to 5.

    Returns:
        pandas DataFrame: Historical transactions and rating
    """
    user = model.train_set.uid_map[userid]
    return list(hist_df[hist_df.useridx == user]["book_id"])[:k]


def get_images(ids, comics_df):
    images = []
    titles = get_title_info(comics_df, [str(i) for i in ids])
    for i, j in list(zip(ids, titles)):
        try:
           img = st.image("../datasets/book_covers/" + str(i) + ".jpg",
                          caption=j, width=200)
           images.append(img)
        except:
            pass
    return images


@st.cache(allow_output_mutation=True)
def load_data():
    comics_df = load_object("assets/book_titles_mapping.pkl")
    model = load_object("assets/model_weights.pkl")
    dataset_map = load_object("assets/dict_map.pkl")
    historical = load_object("assets/historical_subset.pkl")
    return comics_df, model, dataset_map, historical


comics_df, model, dataset_map, historical = load_data()
model.train_set = dataset_map
all_users = list(model.train_set.uid_map.keys())

if __name__ == "__main__":
    # Sidebar Recommendation settings
    st.sidebar.header("Recommendations Settings")
    results = st.sidebar.empty()
    user = st.sidebar.text_input("Please select user_id:")
    st.sidebar.text("Alternatively:")
    random = st.sidebar.button("Generate random user recommendation")
    to_display = st.sidebar.slider(
        "No of recommendations to display",
        min_value=0,
        max_value=50,
        value=10
    )
    show_images = st.sidebar.checkbox("Show images?")

    # Set styles
    select_block_container_style()

    st.image("/Users/davidchong/Desktop/banner 2.jpg", width=1370)
    st.write("## 📚 **GoodReads** Comic *Recommendations*")

    # Grid starts here
    with Grid("1 1 1", color=COLOR, background_color=BACKGROUND_COLOR) as grid:
        grid.cell(
            class_="a",
            grid_column_start=2,
            grid_column_end=4,
            grid_row_start=1,
            grid_row_end=2,
        ).markdown("# Welcome back, <u>*" + RANDOM_NAME + "*</u> 🎲 !")
        grid.cell("b", 2, 3, 2, 3).markdown(
            "## Your reading stats on the left:")
        grid.cell("c", 3, 4, 2, 3).markdown(
            "## Currently Reading: " + str(CURR_READ))
        grid.cell("d", 1, 2, 1, 3).dataframe(pd.DataFrame({
            "Count": [READ_COUNT1, READ_COUNT2, READ_COUNT3]
        }, index=["Books read in the last month",
                  "Books read in the last year",
                  "Books read in the last 5 years"]))
        grid.cell("f", 1, 3, 3, 4).markdown(
            "### What do you want to read *next*?"
        )
        grid.cell("g", 3, 4, 3, 4).markdown(
            "#### Improve Recommendations?<br> Rate more!")

    # Recommendations
    ""
    if random:
        "### Recommedations for " + RANDOM_NAME
        random_user = np.random.choice(all_users)
        results.success(
            "Getting recommendations for user " + str(random_user))
        recs, ids = get_recommendations(model, random_user, comics_df,
                                        k=to_display)
        df = pd.DataFrame({
            "ID": np.arange(1, len(recs)+1),
            "Recommendations": recs
        })
        txn_hist = get_historical_txn(model, random_user, historical, k=5)
        st.table(df.set_index("ID"))
        if show_images:
            images = get_images(ids, comics_df)

    elif user:
        if user in all_users:
            results.success(
                "Getting recommendations for user " + str(user))
            recs, ids = get_recommendations(model, user, comics_df, k=to_display)
            df = pd.DataFrame({
                "ID": np.arange(1, len(recs)+1),
                "Recommendations": recs
            })
            txn_hist = get_historical_txn(model, user, historical, k=5)
            st.table(df.set_index("ID"))
            if show_images:
                images = get_images(ids, comics_df)
        else:
            results.warning("Invalid user!")
    else:
        results.info("You have not selected a user_id!")

    # Historical Interactions
    if random or user:
        "### Because you read:"
        txn_df = pd.DataFrame({
            "ID": np.arange(1, len(txn_hist)+1),
            "Title": get_title_info(comics_df, txn_hist)
        })
        st.table(txn_df.set_index("ID"))
        if show_images:
            hist_images = get_images(txn_hist, comics_df)
