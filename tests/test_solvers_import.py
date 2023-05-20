def test_all_solvers_can_be_imported():
    from solvers.cli import CliGuesser, CliHinter  # noqa
    from solvers.gpt import GPTGuesser, GPTHinter, GPTPlayer  # noqa
    from solvers.naive import (  # noqa
        ModelAwareCliGuesser,
        NaiveGuesser,
        NaiveHinter,
        NaivePlayer,
    )
    from solvers.sna import SNAHinter  # noqa
