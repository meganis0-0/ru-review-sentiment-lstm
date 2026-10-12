from src.data.rule_label import rule_label


def test_rule_reads_complaint_praise_and_bare_fact() -> None:
    assert rule_label("Ужасное хамство и грязь в зале") == "negative"
    assert rule_label("Отличный банк, очень рекомендую") == "positive"
    assert rule_label("Доставили заказ вчера утром") == "neutral"


def test_rule_flips_a_negated_word_and_ties_stay_neutral() -> None:
    assert rule_label("Фильм не плохой") == "positive"
    assert rule_label("Хороший сервис, но ужасный зал") == "neutral"
