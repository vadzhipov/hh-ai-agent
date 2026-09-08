from pathlib import Path

import pytest

from instance_lock import AlreadyRunningError, single_instance


def test_single_instance_rejects_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "agent.db.lock"
    with single_instance(lock_path):
        with pytest.raises(AlreadyRunningError):
            with single_instance(lock_path):
                pass

    with single_instance(lock_path):
        assert lock_path.read_text(encoding="utf-8").strip().isdigit()
