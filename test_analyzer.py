from analyzer import LogReport, analyze_logs
import pytest


@pytest.mark.live_llm
def test_extracts_errors_and_warnings():
    logs = """
    2026-11-23 10:15:02 ERROR auth-service: JWT token expired for user_id=884
    2026-11-23 10:15:09 WARN payment-gateway: Timeout 1500ms while calling bank API
    2026-11-23 10:15:12 INFO notification-service: Email queued successfully
    """
    result = analyze_logs(logs)

    assert isinstance(result, LogReport)
    assert len(result.issues) == 2
    assert "**" not in result.summary  # Markdown жоқ екенін тексеру


@pytest.mark.live_llm
def test_clean_logs_returns_empty_issues():
    clean_logs = """
    2026-11-23 10:15:12 INFO notification-service: Email queued successfully
    2026-11-23 10:15:15 INFO user-service: Profile updated for user_id=102
    """
    result = analyze_logs(clean_logs)

    assert isinstance(result, LogReport)
    assert result.issues == []