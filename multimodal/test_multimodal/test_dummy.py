from health_multimodal.dummy import dummy


def test_dummy() -> None:
    assert dummy() == 1
