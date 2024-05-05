pipeline_config = [

    ("p1", "RobustScaler", {"with_centering": False}, ("age", "fare")),
    ("p1", "SimpleImputer", {"strategy":"median"}, ("age", "fare")),
    ("p2", "OneHotEncoder", {"handle_unknown": "ignore"}, ("pclass", "sex", "sibsp", "parch")),
    ("e1", "LinearRegression", {})
]