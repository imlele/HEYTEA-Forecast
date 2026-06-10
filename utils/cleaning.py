def clean_daily_sku_sales_df(df: pd.DataFrame, sale_date, source_file: str) -> pd.DataFrame:
    """
    Final upload format:
    sale_date, rank, product_name, qty, sales_share, raw_text, source_file
    """

    df = df.copy()

    required_columns = [
        "rank",
        "product_name",
        "qty",
        "sales_share",
        "raw_text",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    df["sale_date"] = sale_date
    df["source_file"] = source_file

    df["rank"] = df["rank"].apply(clean_number)
    df["product_name"] = df["product_name"].apply(clean_text)
    df["qty"] = df["qty"].apply(clean_number)
    df["sales_share"] = df["sales_share"].apply(clean_number)
    df["raw_text"] = df["raw_text"].apply(clean_text)

    df["rank"] = df["rank"].apply(lambda x: int(x) if x is not None else None)

    df = df[
        [
            "sale_date",
            "rank",
            "product_name",
            "qty",
            "sales_share",
            "raw_text",
            "source_file",
        ]
    ]

    df = df.dropna(subset=["sale_date", "product_name", "qty"])

    df = df.drop_duplicates(
        subset=["sale_date", "product_name"],
        keep="last",
    )

    return df.reset_index(drop=True)
