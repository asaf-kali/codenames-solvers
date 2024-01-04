def test_all_solvers_can_be_imported():
    from solvers.cli import CLIGuesser, CLIHinter  # noqa
    from solvers.gpt import GPTGuesser, GPTHinter, GPTPlayer  # noqa
    from solvers.naive import NaiveGuesser, NaiveHinter, NaivePlayer  # noqa
    from solvers.sna import SNAHinter  # noqa
