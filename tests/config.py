test_config = [

    ("p1", "RobustScaler", {"with_centering": False}, ("age", "fare")),
    ("p1", "SimpleImputer", {"strategy":"median"}, ("age", "fare")),
    ("p2", "OneHotEncoder", {"handle_unknown": "ignore"}, ("pclass", "sex", "sibsp", "parch")),
    ("e1", "LinearRegression", {})
]


test_config_2 = [
            ("p1", "RobustScaler", {"with_centering": False}, ("Column1", "Column2")),
            ("p2", "OneHotEncoder", {"handle_unknown": "ignore"}, ("Column3", "Column4")),
            ("e1", "LinearRegression", {})
        ]