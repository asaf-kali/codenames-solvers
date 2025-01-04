from codenames_solvers.models import DefaultFormatAdapter


def test_adapter_uses_init_cache():
    board_to_model = {"foo": "bar"}
    adapter = DefaultFormatAdapter(board_to_model=board_to_model)
    assert adapter.to_board_format("bar") == "foo"
    assert adapter.to_model_format("foo") == "bar"


def test_adapter_finds_variation():
    model_words = {"black_hole", "bigbang"}

    def checker(word: str) -> bool:
        return word in model_words

    adapter = DefaultFormatAdapter(existence_checker=checker)

    assert adapter.to_model_format("black hole") == "black_hole"
    assert adapter.to_model_format("big bang") == "bigbang"
    assert adapter.to_board_format("black_hole") == "black hole"
    assert adapter.to_board_format("bigbang") == "big bang"
