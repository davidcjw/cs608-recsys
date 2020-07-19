import numpy as np
import pandas as pd
import pickle

import cornac
import streamlit as st

def load_object(filename):
    """
    Load a pickled object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)

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

@st.cache(allow_output_mutation=True)
def load_data():
    comics_df = load_object("assets/book_titles_mapping.pkl")
    model = load_object("assets/2020-07-12_12-42-45-810642.pkl")
    dataset_map = load_object("assets/dict_map.pkl")
    historical = load_object("assets/historical_subset.pkl")
    return comics_df, model, dataset_map, historical


comics_df, model, dataset_map, historical = load_data()
model.train_set = dataset_map
all_users = list(model.train_set.uid_map.keys())

if __name__ == "__main__":
    "# GoodReads Recommendations"
    results = st.empty()
    user = st.sidebar.text_input("Please select user_id:")
    random = st.button("Generate random user recommendation")
    to_display = st.slider("No of recommendations to display",
                           min_value=0,
                           max_value=50,
                           value=10)

    if random:
        random_user = np.random.choice(all_users)
        results.success(
            "Getting recommendations for user " + str(random_user))
        recs = get_recommendations(model, random_user, comics_df,
                                   k=to_display)
        df = pd.DataFrame({
            "ID": np.arange(1, len(recs)+1),
            "Recommendations": recs
        })
        txn_hist = get_historical_txn(model, random_user, historical, k=5)
        df

    elif user:
        if user in all_users:
            results.success(
                "Getting recommendations for user " + str(user))
            recs = get_recommendations(model, user, comics_df, k=to_display)
            df = pd.DataFrame({
                "ID": np.arange(1, len(recs)+1),
                "Recommendations": recs
            })
            txn_hist = get_historical_txn(model, user, historical, k=5)
            df
        else:
            results.warning("Invalid user!")
    else:
        results.info("You have not selected a user_id!")

    if random or user:
        "Historical Transactions (displays up to 5)"
        txn_hist = get_title_info(comics_df, [str(i) for i in txn_hist])
        txn_hist_df = pd.DataFrame({
            "ID": np.arange(1, len(txn_hist)+1),
            "Title": txn_hist
        })
        txn_hist_df
