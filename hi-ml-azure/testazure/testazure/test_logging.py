import logging
import os
import pytest
from _pytest.capture import SysCapture
from time import sleep
from health_azure.logging import format_time_from_seconds, elapsed_timer, print_message_with_rank_pid, logging_section
from health_azure.utils import ENV_LOCAL_RANK


def test_format_time_from_second() -> None:
    # Test less than a minute
    time = format_time_from_seconds(0)
    assert time == "0.00 seconds"
    time = format_time_from_seconds(35.5)
    assert time == "35.50 seconds"
    # Test more than a minute
    time = format_time_from_seconds(60)
    assert time == "1.00 minutes"
    time = format_time_from_seconds(60 * 2 + 35.5)
    assert time == "2.59 minutes"
    # Test more than an hour
    time = format_time_from_seconds(60 * 60)
    assert time == "1.00 hours"
    time = format_time_from_seconds(60 * 60 * 2 + 60 * 2 + 35.5)
    assert time == "2.04 hours"


@pytest.mark.parametrize("level", ['DEBUG', 'INFO'])
def test_print_message_with_rank_pid(level: str, capsys: SysCapture) -> None:
    rank = "0"
    os.environ[ENV_LOCAL_RANK] = rank
    message = "test"
    print_message_with_rank_pid(message, level=level)
    stdout: str = capsys.readouterr().out  # type: ignore
    assert f"Rank {rank}" in stdout
    assert "PID" in stdout
    assert message in stdout
    assert level in stdout


@pytest.mark.parametrize("format_seconds", [True, False])
def test_elapsed_timer(format_seconds: bool, capsys: SysCapture) -> None:
    rank = "0"
    os.environ[ENV_LOCAL_RANK] = rank
    with elapsed_timer("test", format_seconds):
        sleep(0.1)  # Sleep for 100 ms
    stdout: str = capsys.readouterr().out  # type: ignore
    assert "0.10 seconds" in stdout


def test_logging_section(caplog: pytest.LogCaptureFixture) -> None:
    rank = "0"
    os.environ[ENV_LOCAL_RANK] = rank
    with logging_section("test"):
        sleep(0.1)  # Sleep for 100 ms
        logging.info("foo")
    messages = caplog.messages  # type: ignore
    assert len(messages) == 7
    assert messages[0] == ""
    assert messages[1].startswith("**** STARTING: test")
    assert messages[2] == ""
    assert messages[3] == "foo"
    assert messages[4] == ""
    assert messages[5].startswith("**** FINISHED: test after 0.10 seconds")
    assert messages[6] == ""
