def test_all_solvers_can_be_imported():
    from solvers.cli.cli_players import CLIOperative, CLISpymaster  # noqa: F401
    from solvers.gpt.gpt_guesser import GPTOperative  # noqa: F401
    from solvers.gpt.gpt_hinter import GPTSpymaster  # noqa: F401
    from solvers.naive.naive_guesser import NaiveOperative  # noqa: F401
    from solvers.naive.naive_hinter import NaiveSpymaster  # noqa: F401
    from solvers.sna.sna_hinter import SNASpymaster  # noqa: F401
