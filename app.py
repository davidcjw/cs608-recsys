import numpy as np

import cornac
import streamlit as st

def get_title_info(comics_df, book_ids: list):
    """
    Attributes:
        comics_df {pandas DataFrame}: dataframe from books_comics_graphic
        book_ids {list}: list of book_id to get title info

    Returns:
        {list}: corresponding titles
    """
    return [comics_df[comics_df.book_id == i].title.values[0]
            for i in book_ids]


def get_recommendations(model, userid, comics_df, k=10):
    """
    Attributes:
        model {instance of trained cornac model}
        userid {int}: p_id

    Returns:
        {list}: top-k recommended items
    """
    user = model.train_set.uid_map[userid]
    recommendations, _ = model.rank(user)

    item_idx2id = list(model.train_set.item_ids)
    top_k_recs = [item_idx2id[i] for i in recommendations[:k]]

    return get_title_info(comics_df, top_k_recs)


# Start of UI
"# Book Recommendations"

user = st.sidebar.text_input("Please select user_id:")

if not user:
    st.warning("You have not selected a user_id!")
else:
    st.success("Getting recommendations for " + str(user))
