def handle_missing_values(df):
    df = df.copy()

    absence_features = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "MasVnrType",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2"
    ]

    for col in absence_features:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    zero_features = [
        "MasVnrArea",
        "GarageCars",
        "GarageArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF"
    ]

    for col in zero_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df