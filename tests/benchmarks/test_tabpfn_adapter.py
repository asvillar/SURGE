from __future__ import annotations


def test_tabpfn_filters_unsupported_kwargs():
    from surge.model.adapters.tabpfn import _filter_supported_kwargs

    class FakeTabPFN:
        def __init__(self, n_estimators=8, softmax_temperature=1.0):
            self.n_estimators = n_estimators
            self.softmax_temperature = softmax_temperature

    filtered = _filter_supported_kwargs(
        FakeTabPFN,
        {
            "n_estimators": 16,
            "softmax_temperature": 0.75,
            "future_only_param": True,
        },
    )
    assert filtered == {"n_estimators": 16, "softmax_temperature": 0.75}


def test_tabpfn_filter_keeps_kwargs_for_variadic_signature():
    from surge.model.adapters.tabpfn import _filter_supported_kwargs

    class VariadicTabPFN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kwargs = {"n_estimators": 16, "future_only_param": True}
    assert _filter_supported_kwargs(VariadicTabPFN, kwargs) == kwargs
