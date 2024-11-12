def test_all_solvers_can_be_imported():
    from solvers.cli.cli_players import CLIOperative, CLISpymaster  # noqa: F401
    from solvers.gpt.gpt_operative import GPTOperative  # noqa: F401
    from solvers.gpt.gpt_spymaster import GPTSpymaster  # noqa: F401
    from solvers.naive.naive_operative import NaiveOperative  # noqa: F401
    from solvers.naive.naive_spymaster import NaiveSpymaster  # noqa: F401
    from solvers.sna.sna_spymaster import SNASpymaster  # noqa: F401
