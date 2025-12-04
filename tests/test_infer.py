from src.infer import predict_weight


def test_predict_weight_basic():
    weight = predict_weight(23.2, 25.4, 30.0, 11.52, 4.02)
    assert isinstance(weight, float)
    assert weight > 0


def test_predict_weight_large_fish():
    weight = predict_weight(40.0, 42.0, 45.0, 20.0, 8.0)
    assert isinstance(weight, float)
    assert weight > 0

def test_predict_weight_medium_fish():
    weight = predict_weight(23.2, 25.4, 30.0, 11.52, 4.02)
    assert isinstance(weight, float)
    assert weight > 0


def test_predict_weight_large_fish():
    weight = predict_weight(35.0, 38.0, 40.0, 15.0, 6.5)
    assert isinstance(weight, float)
    assert weight > 0