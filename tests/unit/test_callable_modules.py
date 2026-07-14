"""Tests to ensure that modules with function name collisions are callable."""


def test_mcmc_output_is_callable():
    # Force import of the submodule to trigger the collision
    import allesfitter
    from allesfitter.mcmc_output import mcmc_output as func

    # Both the imported function and the module object on the package should be callable
    assert callable(func)
    assert callable(allesfitter.mcmc_output)


def test_optimize_is_callable():
    # Force import of the submodule to trigger the collision
    import allesfitter
    from allesfitter.optimize import optimize as func

    # Both should be callable
    assert callable(func)
    assert callable(allesfitter.optimize)


def test_prepare_ttv_fit_is_callable():
    # Force import of the submodule to trigger the collision
    import allesfitter
    from allesfitter.prepare_ttv_fit import prepare_ttv_fit as func

    # Both should be callable
    assert callable(func)
    assert callable(allesfitter.prepare_ttv_fit)
