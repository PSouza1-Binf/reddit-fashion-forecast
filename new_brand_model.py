# new_brand_model.py
"""
Clean, importable version of the forecaster pipeline.
Provides build_model_bundle(posts_path, comments_path, *, do_downloads=False)
which returns a dict with keys:
  - agg, watchlist, pr_auc, mae, surge_threshold, clf, reg, feat, micro, df
Notes:
 - This file no longer contains any Colab or shell commands.
 - If spaCy model or NLTK data are missing, set do_downloads=True to attempt to download.
"""

import re
import pickle
from typing import Optional, Dict, Any, List
from collections import Counter

import numpy as np
import pandas as pd

# ML & NLP
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

# Config
SURGE_TOP_PCT = 0.10
MIN_POSTS_PER_MICROTOPIC = 4
RANDOM_STATE = 42


# -----------------------
# Helpers
# -----------------------
import zipfile

def load_csv_from_zip(path):
    """Load the first CSV found inside a ZIP archive."""
    with zipfile.ZipFile(path, "r") as z:
        csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def _ensure_nltk_vader(do_download: bool = False):
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        if do_download:
            import nltk
            nltk.download("vader_lexicon")
            return SentimentIntensityAnalyzer()
        else:
            raise RuntimeError(
                "NLTK VADER lexicon not found. Either pre-install it (python -m nltk.downloader vader_lexicon) "
                "or call build_model_bundle(..., do_downloads=True)."
            )
    return SentimentIntensityAnalyzer()


def _ensure_spacy_model(do_download: bool = False):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        return nlp
    except Exception:
        if do_download:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
            return nlp
        else:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not installed. Install it or call build_model_bundle(..., do_downloads=True)."
            )


def norm(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())


# -----------------------
# Brand discovery & regex builders
# -----------------------
def discover_brands_spacy_on_titles(titles: List[str], nlp, min_count: int = 10) -> pd.DataFrame:
    counter = Counter()
    # Use large batch, single process (stable on many environments)
    for doc in nlp.pipe(titles, batch_size=512, n_process=1):
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT"}:
                name = ent.text.strip()
                if len(name) >= 2:
                    counter[name] += 1
    cand = pd.DataFrame([{"brand": k, "count": v} for k, v in counter.items()]).sort_values("count", ascending=False)
    return cand[cand["count"] >= min_count].reset_index(drop=True)


def build_brand_item_regex_from_dataframe(df: pd.DataFrame):
    titles = df["title"].fillna("").tolist()
    nlp = _ensure_spacy_model(do_download=False)
    brand_candidates = discover_brands_spacy_on_titles(titles, nlp, min_count=10)

    blacklist = {"Today", "Monday", "Tuesday", "Friday", "Reddit", "YouTube", "Instagram", "WhatsApp", "Buy", "Need"}
    auto_brands = [b for b in brand_candidates["brand"].tolist() if b not in blacklist]

    manual_additions = ["Nike", "Adidas", "Jordan", "New Balance", "Yeezy", "Stüssy", "Aimé Leon Dore"]
    BRANDS = sorted({b.strip() for b in auto_brands + manual_additions if b and b.strip()})

    ITEMS = [
        "Dunk", "Dunks", "Air Force 1", "AF1", "Jordan 1", "Jordan 4", "Jordan 3",
        "Samba", "Gazelle", "Campus", "Huarache",
        "hoodie", "hoodies", "crewneck", "t-shirt", "tee", "tees",
        "cargo pants", "cargos", "jeans", "denim", "puffer", "parka",
        "tracksuit", "track jacket", "track pants",
        "bag", "tote bag", "backpack", "belt", "cap", "hat", "beanie",
    ]

    def _esc(s): return re.escape(s)
    brand_pattern = r"|".join(sorted(_esc(b) for b in BRANDS)) if BRANDS else ""
    item_pattern = r"|".join(sorted(_esc(i) for i in ITEMS)) if ITEMS else ""

    BRAND_RE = re.compile(rf"\b({brand_pattern})\b", flags=re.IGNORECASE) if brand_pattern else None
    ITEM_RE = re.compile(rf"\b({item_pattern})\b", flags=re.IGNORECASE) if item_pattern else None

    return BRANDS, ITEMS, BRAND_RE, ITEM_RE


# -----------------------
# Core pipeline
# -----------------------
def build_model_bundle(posts_path: str,
                       comments_path: str,
                       *,
                       do_downloads: bool = False) -> Dict[str, Any]:
    """
    Build the full pipeline from raw CSV(s) and return a bundle dict.
    Set do_downloads=True to allow the function to download missing NLP assets (not recommended for Streamlit).
    """
    # Ensure resources
    vader = _ensure_nltk_vader(do_download=do_downloads)
    nlp = _ensure_spacy_model(do_download=do_downloads)

    # Load zipped data files (located in /Data/)
    df = load_csv_from_zip("Data/merged_reddit_posts_final.zip")
    comments_df = load_csv_from_zip("Data/all_comments_multi2.zip")


    # --- clean posts ---
    posts["title"] = posts["title"].apply(norm)
    posts["selftext"] = posts.get("selftext", "").apply(norm)

    # normalize name fields for merging
    posts["name"] = posts["name"].astype(str).str.replace(r"^t3_", "", regex=True)
    comments["link_id"] = comments["link_id"].astype(str).str.replace(r"^t3_", "", regex=True)
    comments["body"] = comments["body"].astype(str).apply(norm)

    # aggregate comments
    comments_per_post = (
        comments.groupby("link_id")["body"]
        .apply(lambda texts: " ".join(t for t in texts if isinstance(t, str)))
        .reset_index()
        .rename(columns={"link_id": "name", "body": "comments_text"})
    )

    if "comments_text" in posts.columns:
        posts = posts.drop(columns=["comments_text"])

    df = posts.merge(comments_per_post, on="name", how="left")
    df["comments_text"] = df["comments_text"].fillna("")
    df["all_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("") + " " + df["comments_text"].fillna("")).str.strip()

    # datetimes + engagement
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
    df = df.dropna(subset=["created_utc"]).copy()
    df["score"] = df["score"].astype(float)
    df["num_comments"] = df["num_comments"].astype(float)
    df["engagement"] = df["score"] + df["num_comments"]

    # sentiment
    df["sentiment"] = df["all_text"].apply(lambda t: float(vader.polarity_scores(t)["compound"]) if isinstance(t, str) else 0.0)

    # brand/item discovery (uses spaCy NER on titles for speed)
    BRANDS, ITEMS, BRAND_RE, ITEM_RE = build_brand_item_regex_from_dataframe(df)

    # extraction helpers
    def canonicalize_brand(raw: str) -> str:
        s = re.sub(r"\s+", " ", raw.strip().lower())
        if s in {"nb", "new balance"}:
            return "new balance"
        return s

    def canonicalize_item(raw: str) -> str:
        s = re.sub(r"\s+", " ", raw.strip().lower())
        if s in {"af1", "af1s", "air force 1", "air force 1s", "forces"}:
            return "air force 1"
        return s

    def extract_brands(text):
        if BRAND_RE is None or not isinstance(text, str):
            return []
        return sorted({canonicalize_brand(m.group(0)) for m in BRAND_RE.finditer(text)})

    def extract_items(text):
        if ITEM_RE is None or not isinstance(text, str):
            return []
        return sorted({canonicalize_item(m.group(0)) for m in ITEM_RE.finditer(text)})

    df["brands"] = df["all_text"].apply(extract_brands)
    df["items"] = df["all_text"].apply(extract_items)

    # build microtopics
    rows = []
    for _, r in df.iterrows():
        brands = r["brands"]
        items = r["items"]
        if not brands:
            continue
        pairs = [(b, "GEN") for b in brands] if not items else [(b, it) for b in brands for it in items]
        pairs = pairs[:6]
        week_start = r["created_utc"].to_period("W").start_time
        for b, it in pairs:
            rows.append({
                "id": r["id"],
                "week_start": week_start,
                "iso_year": int(r["created_utc"].isocalendar().year),
                "iso_week": int(r["created_utc"].isocalendar().week),
                "brand": b,
                "item": it,
                "microtopic": f"{b} | {it}",
                "engagement": r["engagement"],
                "score": r["score"],
                "num_comments": r["num_comments"],
                "sentiment": r["sentiment"],
                "has_selftext": 1.0 if len(r.get("selftext", "")) > 0 else 0.0,
            })
    micro = pd.DataFrame(rows)
    if micro.empty:
        raise RuntimeError("No microtopics found after extraction. Check your BRANDS/ITEMS patterns and data.")

    # weekly aggregation
    agg = (micro.groupby(["microtopic", "brand", "item", "iso_year", "iso_week", "week_start"], as_index=False)
           .agg(posts=("id", "count"),
                engagement_sum=("engagement", "sum"),
                score_sum=("score", "sum"),
                comments_sum=("num_comments", "sum"),
                sentiment_mean=("sentiment", "mean"),
                selftext_rate=("has_selftext", "mean")))
    # filter small topics
    topic_sizes = agg.groupby("microtopic")["posts"].sum().reset_index(name="total_posts")
    big_topics = topic_sizes[topic_sizes["total_posts"] >= MIN_POSTS_PER_MICROTOPIC]["microtopic"]
    agg = agg[agg["microtopic"].isin(big_topics)].reset_index(drop=True)

    # rolling features per microtopic
    def add_rolling_features(g):
        g = g.sort_values(["iso_year", "iso_week"]).copy()
        g["engagement_prev"] = g["engagement_sum"].shift(1).fillna(0.0)
        g["engagement_wow"] = (g["engagement_sum"] - g["engagement_prev"]) / g["engagement_prev"].replace(0, 1.0)
        g["engagement_ema3"] = g["engagement_sum"].ewm(span=3, adjust=False).mean()
        g["engagement_ema7"] = g["engagement_sum"].ewm(span=7, adjust=False).mean()
        g["posts_prev"] = g["posts"].shift(1).fillna(0.0)
        g["posts_wow"] = (g["posts"] - g["posts_prev"]) / g["posts_prev"].replace(0, 1.0)
        g["sentiment_prev"] = g["sentiment_mean"].shift(1).fillna(0.0)
        g["sentiment_delta"] = g["sentiment_mean"] - g["sentiment_prev"]
        g["engagement_next"] = g["engagement_sum"].shift(-1)
        g["growth_next"] = (g["engagement_next"] - g["engagement_sum"]) / g["engagement_sum"].replace(0, 1.0)
        return g

    feat = agg.groupby("microtopic", group_keys=False).apply(add_rolling_features)
    feat["weekofyear_sin"] = np.sin(2 * np.pi * feat["iso_week"] / 52.0)
    feat["weekofyear_cos"] = np.cos(2 * np.pi * feat["iso_week"] / 52.0)
    feat = feat.dropna(subset=["engagement_next"]).reset_index(drop=True)

    # train/test split by time
    all_weeks = (feat[["iso_year", "iso_week", "week_start"]]
                 .drop_duplicates()
                 .sort_values(["iso_year", "iso_week"])
                 .reset_index(drop=True))
    split_idx = int(len(all_weeks) * 0.70)
    split_year = int(all_weeks.loc[split_idx, "iso_year"])
    split_week = int(all_weeks.loc[split_idx, "iso_week"])

    def is_train(yr, wk):
        return (yr < split_year) or (yr == split_year and wk <= split_week)

    feat["is_train"] = feat.apply(lambda r: is_train(int(r["iso_year"]), int(r["iso_week"])), axis=1)
    train_df = feat[feat["is_train"]].copy()
    test_df = feat[~feat["is_train"]].copy()

    surge_thresh = float(np.quantile(train_df["growth_next"].dropna(), 1.0 - SURGE_TOP_PCT))
    train_df["y_cls"] = (train_df["growth_next"] >= surge_thresh).astype(int)
    test_df["y_cls"] = (test_df["growth_next"] >= surge_thresh).astype(int)

    FEATURE_COLS = [
        "posts", "engagement_sum", "score_sum", "comments_sum",
        "sentiment_mean", "selftext_rate",
        "engagement_prev", "engagement_wow", "engagement_ema3", "engagement_ema7",
        "posts_prev", "posts_wow",
        "sentiment_prev", "sentiment_delta",
        "weekofyear_sin", "weekofyear_cos",
    ]

    def safe_X(df_in: pd.DataFrame) -> pd.DataFrame:
        X = df_in[FEATURE_COLS].copy()
        return X.fillna(0.0).astype(float)

    X_train = safe_X(train_df)
    y_train_cls = train_df["y_cls"].astype(int)
    y_train_reg = train_df["engagement_next"].astype(float)
    X_test = safe_X(test_df)
    y_test_cls = test_df["y_cls"].astype(int)
    y_test_reg = test_df["engagement_next"].astype(float)

    # Train models
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train_cls)

    reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    reg.fit(X_train, y_train_reg)

    # Evaluate
    probs = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test_cls, probs)
    pred_next = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, pred_next)

    # Watchlist (latest week)
    latest_year = int(feat["iso_year"].max())
    latest_week = int(feat[feat["iso_year"] == latest_year]["iso_week"].max())
    latest = feat[(feat["iso_year"] == latest_year) & (feat["iso_week"] == latest_week)].copy()
    X_latest = safe_X(latest)
    latest["surge_prob"] = clf.predict_proba(X_latest)[:, 1]
    latest["pred_next_engagement"] = reg.predict(X_latest)
    watchlist = latest[
        ["microtopic", "brand", "item", "iso_year", "iso_week", "week_start",
         "posts", "engagement_sum", "sentiment_mean", "surge_prob", "pred_next_engagement"]
    ].sort_values(["surge_prob", "pred_next_engagement"], ascending=False).reset_index(drop=True)

    return {
        "agg": agg,
        "watchlist": watchlist,
        "pr_auc": pr_auc,
        "mae": mae,
        "surge_threshold": surge_thresh,
        "clf": clf,
        "reg": reg,
        "feat": feat,
        "micro": micro,
        "df": df,
    }


# Optional helper to save a pickle bundle
def save_model_bundle(posts_path: str, comments_path: str, out_path: str = "model_bundle.pkl", do_downloads: bool = False):
    bundle = build_model_bundle(posts_path, comments_path, do_downloads=do_downloads)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model bundle to {out_path}")


# If executed directly, build and save a pickle (useful for offline runs)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and optionally save model bundle.")
    parser.add_argument("--posts", default="merged_reddit_posts_final.zip", help="Path to posts CSV or zip")
    parser.add_argument("--comments", default="all_comments_multi2.zip", help="Path to comments CSV or zip")
    parser.add_argument("--out", default="model_bundle.pkl", help="Output pickle path")
    parser.add_argument("--download", action="store_true", help="Allow auto-download of missing NLP assets (not recommended in production)")
    args = parser.parse_args()
    save_model_bundle(args.posts, args.comments, args.out, do_downloads=args.download)

