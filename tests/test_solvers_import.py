def test_all_solvers_can_be_imported():
    from codenames_solvers.cli.cli_players import (  # noqa: F401
        CLIOperative,
        CLISpymaster,
    )
    from codenames_solvers.gpt.gpt_operative import GPTOperative  # noqa: F401
    from codenames_solvers.gpt.gpt_spymaster import GPTSpymaster  # noqa: F401
    from codenames_solvers.naive.naive_operative import NaiveOperative  # noqa: F401
    from codenames_solvers.naive.naive_spymaster import NaiveSpymaster  # noqa: F401
    from codenames_solvers.sna.sna_spymaster import SNASpymaster  # noqa: F401
