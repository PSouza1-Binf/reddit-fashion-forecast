import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
SURGE_TOP_PCT = 0.10          # top 10% growth = "surge"
MIN_POSTS_PER_MICROTOPIC = 4  # filter out tiny topics
RANDOM_STATE = 42


# -------------------------------------------------------------------
# BASIC PREP – MATCHES YOUR CSV
# -------------------------------------------------------------------
def norm(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())


def basic_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw Reddit post-level data.

    Works with columns like:
    ['id', 'name', 'title', 'author', 'engagement', 'num_comments',
     'created_utc', 'url', 'permalink', 'upvote_ratio', 'selftext', 'subreddit']
    """
    df = df.copy()

    # Ensure needed columns exist
    if "title" not in df.columns:
        df["title"] = ""
    if "selftext" not in df.columns:
        df["selftext"] = ""
    if "num_comments" not in df.columns:
        df["num_comments"] = 0.0

    # If engagement is present, use it; else compute from score + num_comments
    if "engagement" not in df.columns:
        if "score" in df.columns:
            df["engagement"] = df["score"].astype(float) + df["num_comments"].astype(float)
        else:
            df["engagement"] = df["num_comments"].astype(float)

    # If score is missing, mirror engagement so we can still use it as a feature
    if "score" not in df.columns:
        df["score"] = df["engagement"].astype(float)

    df["title"] = df["title"].apply(norm)
    df["selftext"] = df["selftext"].apply(norm)
    df["all_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()

    # created_utc: unix seconds, as in your CSV
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
    elif "created" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    else:
        df["created_utc"] = pd.NaT

    df = df.dropna(subset=["created_utc"]).copy()
    # engagement already defined above
    return df


# -------------------------------------------------------------------
# SENTIMENT (very simple heuristic)
# -------------------------------------------------------------------
POSITIVE = {
    # General positivity
    "love": 2.0, "like": 1.0, "great": 1.5, "amazing": 2.0, "fire": 2.0,
    "clean": 1.2, "dope": 1.5, "nice": 1.0, "awesome": 2.0, "beautiful": 1.6,
    "perfect": 2.0, "solid": 1.0, "good": 1.0, "heat": 1.8, "grail": 2.0,
    "legit": 1.2, "quality": 1.2, "cop": 1.2, "copped": 1.5, "w": 1.2, "win": 1.5,
    # Added slang / community language
    "clean af": 2.0, "super clean": 2.0, "gas": 1.8, "firee": 1.8, "insane": 1.5,
    "crazy": 1.2, "vintage": 1.0, "cozy": 1.2, "slept": 1.5, "underrated": 1.3,
    "comfortable": 1.2, "fits": 1.2, "heat": 1.8, "essential": 1.2, "banger": 1.8,
    # Approval / satisfaction
    "worth": 1.0, "recommend": 1.5, "satisfied": 1.5, "happy": 1.5, "love it": 2.0,
    "amazing quality": 2.0, "best": 2.0, "beautiful": 1.6, "wow": 1.5,
    # Trend and hype words
    "hyped": 1.2, "trend": 1.0, "viral": 1.2, "drippy": 1.8, "sleek": 1.2,
    "clean fit": 1.8, "solid pickup": 1.5, "good deal": 1.3, "steal": 1.2,
    "🔥": 2.0, "💯": 1.8, "👌": 1.2
}
NEGATIVE = {
    # General negativity
    "hate": -2.0, "dislike": -1.2, "trash": -2.0, "ugly": -1.5, "bad": -1.2,
    "fake": -2.0, "rep": -0.8, "reps": -0.8, "mid": -1.0, "overpriced": -1.5,
    "cheap": -0.8, "worse": -1.0, "awful": -2.0, "terrible": -2.0, "l": -1.2, "lose": -1.2,
    # Added slang / community language
    "mid af": -1.5, "trash af": -2.0, "poor": -1.0, "bad quality": -1.5,
    "uncomfortable": -1.2, "ugly af": -2.0, "hate it": -2.0, "broke": -1.2,
    "fake qc": -1.5, "flawed": -1.0, "stiff": -0.8, "creases": -0.8,
    # Negative sentiment around pricing / value
    "scam": -2.0, "overhyped": -1.5, "waste": -1.5, "bad deal": -1.2,
    "returning": -1.0, "refund": -1.0, "expensive": -0.8,
    # Emojis / shorthand
    "💀": -1.5, "🤮": -2.0, "👎": -1.5, "😭": -1.2, "😤": -1.2, "😡": -1.5
}



def simple_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    score = 0.0
    for w, v in POSITIVE.items():
        if w in t:
            score += v
    for w, v in NEGATIVE.items():
        if w in t:
            score += v
    # squash into [-1, 1]
    return float(np.tanh(score / 3.0))


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["all_text"].apply(simple_sentiment)
    return df


# -------------------------------------------------------------------
# BRAND & ITEM EXTRACTION (lexicon-based)
# -------------------------------------------------------------------
BRANDS = [
    # Sneaker & streetwear core
    "Nike", "Adidas", "Jordan", "New Balance", "Puma", "Reebok", "ASICS", "Vans",
    "Converse", "Yeezy", "Supreme", "Palace", "BAPE", "Kith", "Stussy", "Fear of God",
    "Essentials", "Chrome Hearts", "Aime Leon Dore", "Noah", "Off-White", "Rhude",
    "Carhartt", "Dickies", "Levi's", "Wrangler",
    # Luxury / designer
    "Gucci", "Prada", "Balenciaga", "Louis Vuitton", "Chanel", "Hermes", "Dior",
    "Burberry", "Versace", "Fendi", "Givenchy", "Alexander McQueen", "Maison Margiela",
    "Rick Owens", "Raf Simons", "Jil Sander", "Dries Van Noten", "AMI Paris", "Loewe",
    "Celine", "Valentino", "Moschino", "Lanvin", "Kenzo",
    # Outdoor / techwear / performance
    "Arc'teryx", "The North Face", "TNF", "Patagonia", "Salomon", "Columbia",
    "Helly Hansen", "Mammut", "Moncler", "Stone Island", "CP Company", "Napapijri",
    # Fast-fashion / accessible
    "Uniqlo", "Zara", "H&M", "Bershka", "Pull & Bear", "COS", "Weekday",
    # Footwear / high-end sneakers
    "Dr. Martens", "Doc Martens", "Docs", "Timberland", "Clarks", "UGG", "Birkenstock",
    "Common Projects", "Onitsuka Tiger", "Veja", "Saucony", "HOKA", "Crocs",
    # Jewelry / accessories
    "Tiffany", "Cartier", "Rolex", "Casio", "G-Shock", "Tag Heuer", "Omega",
    "Swarovski", "Pandora", "Sun House", "Vivienne Westwood",
    # Misc / emerging labels seen in reps & reddit culture
    "Aritzia", "COSRX", "Muji", "Acne Studios", "Our Legacy", "Needles",
    "Kapital", "John Elliott", "Aimé Leon Dore", "Y-3", "Reformation", "Aritzia",
    "Represent", "1017 ALYX 9SM", "Fear of God Essentials", "Heron Preston",
    "AMBUSH", "Daily Paper", "Arcteryx", "Maison Mihara", "Mihara Yasuhiro"
]

ITEMS = [
    # Footwear
    "Dunk", "Dunks", "Jordan 1", "Jordan 3", "Jordan 4", "Jordan 11",
    "Air Force 1", "AF1", "Air Max", "Samba", "Gazelle", "Campus", "Superstar",
    "Stan Smith", "Forum", "550", "574", "990", "991", "992", "993", "2002R",
    "Tabi", "Ramones", "Boots", "Loafers", "Derby", "Clogs", "Slides", "Sandals",
    "Sneakers", "Runners", "Crocs","Shoes","Sneakers",
    # Tops / outerwear
    "Hoodie", "Crewneck", "Sweater", "Cardigan", "Jacket", "Denim Jacket",
    "Bomber", "Varsity Jacket", "Puffer", "Down Jacket", "Fleece", "Parka",
    "Shell", "Windbreaker", "Raincoat", "Vest", "Blazer", "Coat", "Trench",
    # Bottoms
    "Jeans", "Denim", "Trousers", "Pants", "Cargo", "Carpenter", "Chinos",
    "Shorts", "Skirt", "Leggings", "Joggers", "Trackpants",
    # Shirts / tees
    "Tee", "T-shirt", "Graphic Tee", "Long Sleeve", "Polo", "Button-up",
    "Flannel", "Henley",
    # Accessories
    "Cap", "Hat", "Beanie", "Scarf", "Belt", "Gloves", "Socks", "Bag", "Tote",
    "Backpack", "Messenger", "Wallet", "Bracelet", "Ring", "Necklace", "Watch",
    "Sunglasses", "Glasses",

]


def compile_lexicon_regex(words):
    escaped = [re.escape(w) for w in words]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


BRAND_RE = compile_lexicon_regex(BRANDS)
ITEM_RE = compile_lexicon_regex(ITEMS)


def extract_brands(text: str):
    if not isinstance(text, str):
        return []
    return sorted({m.group(0).strip() for m in BRAND_RE.finditer(text)})


def extract_items(text: str):
    if not isinstance(text, str):
        return []
    return sorted({m.group(0).strip() for m in ITEM_RE.finditer(text)})


def add_brand_item_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brands"] = df["all_text"].str.lower().apply(extract_brands)
    df["items"] = df["all_text"].str.lower().apply(extract_items)
    return df


# -------------------------------------------------------------------
# MICROTPOICS + WEEKLY AGGREGATION
# -------------------------------------------------------------------
def build_micro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert post-level data into microtopic weekly rows:
    one row = (brand, item, week).
    """
    rows = []
    for _, r in df.iterrows():
        brands = r["brands"]
        items = r["items"]
        if not brands:
            continue
        if not items:
            pairs = [(b, "gen") for b in brands]   # generic if no item
        else:
            pairs = [(b, it) for b in brands for it in items]
        pairs = pairs[:6]  # cap for safety
        for b, it in pairs:
            rows.append({
                "id": r.get("id", None),
                "week_start": r["created_utc"].to_period("W").start_time,
                "iso_year": int(r["created_utc"].isocalendar().year),
                "iso_week": int(r["created_utc"].isocalendar().week),
                "brand": b,
                "item": it,
                "microtopic": f"{b} | {it}",
                "engagement": r["engagement"],
                "score": r["score"],
                "num_comments": r["num_comments"],
                "sentiment": r["sentiment"],
                "has_selftext": 1.0 if len(str(r["selftext"])) > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def build_agg(micro: pd.DataFrame) -> pd.DataFrame:
    if micro.empty:
        return micro
    agg = (micro.groupby(["microtopic", "brand", "item", "iso_year", "iso_week", "week_start"])
                 .agg(
                     posts=("id", "count"),
                     engagement_sum=("engagement", "sum"),
                     score_sum=("score", "sum"),
                     comments_sum=("num_comments", "sum"),
                     sentiment_mean=("sentiment", "mean"),
                     selftext_rate=("has_selftext", "mean"),
                 )
                 .reset_index())
    topic_sizes = agg.groupby("microtopic")["posts"].sum()
    big_topics = topic_sizes[topic_sizes >= MIN_POSTS_PER_MICROTOPIC].index
    agg = agg[agg["microtopic"].isin(big_topics)].copy()
    agg = agg.sort_values(["microtopic", "iso_year", "iso_week"]).reset_index(drop=True)
    return agg


def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
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


def build_feat(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    feat = agg.groupby("microtopic", group_keys=False).apply(add_rolling_features)
    feat["weekofyear_sin"] = np.sin(2 * np.pi * feat["iso_week"] / 52.0)
    feat["weekofyear_cos"] = np.cos(2 * np.pi * feat["iso_week"] / 52.0)
    feat = feat.dropna(subset=["engagement_next"]).reset_index(drop=True)
    return feat


# -------------------------------------------------------------------
# MODELING
# -------------------------------------------------------------------
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


def time_split(feat: pd.DataFrame):
    all_weeks = (feat[["iso_year", "iso_week", "week_start"]]
                 .drop_duplicates()
                 .sort_values(["iso_year", "iso_week"])
                 .reset_index(drop=True))
    if len(all_weeks) < 3:
        feat = feat.copy()
        feat["is_train"] = True
        return feat, feat.iloc[0:0].copy()

    split_idx = int(len(all_weeks) * 0.70)
    split_year = int(all_weeks.loc[split_idx, "iso_year"])
    split_week = int(all_weeks.loc[split_idx, "iso_week"])

    def is_train(yr, wk) -> bool:
        return (yr < split_year) or (yr == split_year and wk <= split_week)

    feat = feat.copy()
    feat["is_train"] = feat.apply(lambda r: is_train(int(r["iso_year"]), int(r["iso_week"])), axis=1)
    train_df = feat[feat["is_train"]].copy()
    test_df = feat[~feat["is_train"]].copy()
    return train_df, test_df


def train_models(feat: pd.DataFrame):
    train_df, test_df = time_split(feat)
    if train_df.empty:
        return None, None, None, float("nan"), float("nan")

    surge_thresh = float(np.quantile(train_df["growth_next"].dropna(), 1.0 - SURGE_TOP_PCT))
    train_df["y_cls"] = (train_df["growth_next"] >= surge_thresh).astype(int)
    if not test_df.empty:
        test_df["y_cls"] = (test_df["growth_next"] >= surge_thresh).astype(int)

    X_train = safe_X(train_df)
    y_train_cls = train_df["y_cls"].astype(int)
    y_train_reg = train_df["engagement_next"].astype(float)

    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train_cls)

    reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    reg.fit(X_train, y_train_reg)

    if not test_df.empty:
        X_test = safe_X(test_df)
        y_test_cls = test_df["y_cls"].astype(int)
        y_test_reg = test_df["engagement_next"].astype(float)
        probs = clf.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test_cls, probs)
        pred_next = reg.predict(X_test)
        mae = mean_absolute_error(y_test_reg, pred_next)
    else:
        pr_auc = float("nan")
        mae = float("nan")

    return clf, reg, surge_thresh, pr_auc, mae


# -------------------------------------------------------------------
# WATCHLIST & PIPELINE
# -------------------------------------------------------------------
def build_watchlist(feat: pd.DataFrame, clf, reg, surge_thresh: float) -> pd.DataFrame:
    if feat.empty or clf is None or reg is None:
        return pd.DataFrame()
    latest_year = int(feat["iso_year"].max())
    latest_week = int(feat[feat["iso_year"] == latest_year]["iso_week"].max())
    latest = feat[(feat["iso_year"] == latest_year) & (feat["iso_week"] == latest_week)].copy()
    if latest.empty:
        return latest

    X_latest = safe_X(latest)
    latest["surge_prob"] = clf.predict_proba(X_latest)[:, 1]
    latest["pred_next_engagement"] = reg.predict(X_latest)

    cols = [
        "microtopic", "brand", "item", "iso_year", "iso_week", "week_start",
        "posts", "engagement_sum", "sentiment_mean",
        "surge_prob", "pred_next_engagement",
    ]
    watchlist = (latest[cols]
                 .sort_values(["surge_prob", "pred_next_engagement"], ascending=False)
                 .reset_index(drop=True))
    return watchlist


def build_everything_from_posts(raw_df: pd.DataFrame):
    """
    End-to-end pipeline used by the Streamlit app.
    It does NOT expect brand/week columns; it infers them.
    """
    df = basic_prepare(raw_df)
    df = add_sentiment(df)
    df = add_brand_item_columns(df)

    micro = build_micro(df)
    agg = build_agg(micro)
    feat = build_feat(agg)

    if feat.empty:
        return {
            "df": df,
            "micro": micro,
            "agg": agg,
            "feat": feat,
            "clf": None,
            "reg": None,
            "surge_thresh": None,
            "pr_auc": float("nan"),
            "mae": float("nan"),
            "watchlist": pd.DataFrame(),
        }

    clf, reg, surge_thresh, pr_auc, mae = train_models(feat)
    watchlist = build_watchlist(feat, clf, reg, surge_thresh)

    return {
        "df": df,
        "micro": micro,
        "agg": agg,
        "feat": feat,
        "clf": clf,
        "reg": reg,
        "surge_thresh": surge_thresh,
        "pr_auc": pr_auc,
        "mae": mae,
        "watchlist": watchlist,
    }
