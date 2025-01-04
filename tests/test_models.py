from codenames_solvers.models import ModelIdentifier


def test_model_identifier():
    mid1 = ModelIdentifier(language="en", model_name="x", is_stemmed=False)
    mid2 = ModelIdentifier(language="en", model_name="x", is_stemmed=False)
    assert mid1 == mid2
    assert hash(mid1) == hash(mid2)
