from src.hello import greet


def test_greet_basic():
    assert greet('World') == 'Hello, World!'
