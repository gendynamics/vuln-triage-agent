#!/usr/bin/env python3
"""Test suite for VulnTriage — schema validation, tier routing, mocked inference."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import SYSTEM_PROMPT, USER_TEMPLATE, MODEL_ROUTING, triage

REQUIRED_KEYS = {
    "severity", "cvss_score", "attack_technique", "summary",
    "remediation_steps", "affected_assets", "estimated_effort", "priority",
}
VALID_SEVERITIES = {"critical", "high", "medium", "low", "informational"}
VALID_EFFORTS = {"S", "M", "L"}


def make_valid_response():
    return {
        "severity": "high",
        "cvss_score": 8.6,
        "attack_technique": "T1190 - Exploit Public-Facing Application",
        "summary": "Unauthenticated SQL injection allows arbitrary query execution.",
        "remediation_steps": [
            "Parameterize all database queries using prepared statements.",
            "Apply input validation and allow-listing on the q parameter.",
            "Deploy a WAF rule blocking SQLi patterns as defense in depth.",
        ],
        "affected_assets": ["production API", "PostgreSQL backend"],
        "estimated_effort": "M",
        "priority": 2,
    }


class TestSchemaValidation(unittest.TestCase):
    def test_valid_response_passes(self):
        resp = make_valid_response()
        self._assert_schema(resp)

    def test_missing_key_rejected(self):
        resp = make_valid_response()
        del resp["cvss_score"]
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def test_invalid_severity_rejected(self):
        resp = make_valid_response()
        resp["severity"] = "extreme"
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def test_cvss_out_of_range_rejected(self):
        resp = make_valid_response()
        resp["cvss_score"] = 11.0
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def test_remediation_steps_must_be_list(self):
        resp = make_valid_response()
        resp["remediation_steps"] = "fix it"
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def test_priority_bounds(self):
        resp = make_valid_response()
        resp["priority"] = 0
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)
        resp["priority"] = 6
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def test_estimated_effort_enum(self):
        resp = make_valid_response()
        resp["estimated_effort"] = "XL"
        with self.assertRaises(AssertionError):
            self._assert_schema(resp)

    def _assert_schema(self, resp):
        missing = REQUIRED_KEYS - set(resp.keys())
        self.assertEqual(missing, set(), f"missing keys: {missing}")
        self.assertIn(resp["severity"], VALID_SEVERITIES)
        self.assertIsInstance(resp["cvss_score"], (int, float))
        self.assertTrue(0.0 <= resp["cvss_score"] <= 10.0)
        self.assertIsInstance(resp["attack_technique"], str)
        self.assertTrue(resp["attack_technique"].startswith("T"))
        self.assertIsInstance(resp["summary"], str)
        self.assertGreater(len(resp["summary"]), 10)
        self.assertIsInstance(resp["remediation_steps"], list)
        self.assertGreaterEqual(len(resp["remediation_steps"]), 1)
        self.assertIsInstance(resp["affected_assets"], list)
        self.assertIn(resp["estimated_effort"], VALID_EFFORTS)
        self.assertIsInstance(resp["priority"], int)
        self.assertTrue(1 <= resp["priority"] <= 5)


class TestModelTierRouting(unittest.TestCase):
    def test_all_tiers_mapped(self):
        self.assertIn("nano", MODEL_ROUTING)
        self.assertIn("super", MODEL_ROUTING)
        self.assertIn("ultra", MODEL_ROUTING)

    def test_nano_is_smallest(self):
        self.assertIn("nano", MODEL_ROUTING["nano"])

    def test_ultra_is_default_fallback(self):
        # unknown tier should fall back to ultra
        with patch("agent._client") as mock_client:
            mock_create = MagicMock()
            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content=json.dumps(make_valid_response())))
            ]
            mock_client.return_value.chat.completions.create = mock_create
            with patch.dict(os.environ, {"NEBIUS_API_KEY": "v1.test"}):
                triage("test report", tier="does_not_exist")
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["model"], MODEL_ROUTING["ultra"])

    def test_nano_tier_selected(self):
        with patch("agent._client") as mock_client:
            mock_create = MagicMock()
            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content=json.dumps(make_valid_response())))
            ]
            mock_client.return_value.chat.completions.create = mock_create
            with patch.dict(os.environ, {"NEBIUS_API_KEY": "v1.test"}):
                triage("test report", tier="nano")
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["model"], MODEL_ROUTING["nano"])

    def test_temperature_is_zero(self):
        with patch("agent._client") as mock_client:
            mock_create = MagicMock()
            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content=json.dumps(make_valid_response())))
            ]
            mock_client.return_value.chat.completions.create = mock_create
            with patch.dict(os.environ, {"NEBIUS_API_KEY": "v1.test"}):
                triage("test report")
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["temperature"], 0.0)


class TestMockedTokenFactory(unittest.TestCase):
    # reuse the schema assertions defined on TestSchemaValidation
    _assert_schema = TestSchemaValidation._assert_schema

    def test_triage_returns_parsed_json(self):
        with patch("agent._client") as mock_client:
            mock_create = MagicMock()
            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content=json.dumps(make_valid_response())))
            ]
            mock_client.return_value.chat.completions.create = mock_create
            with patch.dict(os.environ, {"NEBIUS_API_KEY": "v1.test"}):
                result = triage("SQL injection in login form")
            self._assert_schema(result)
            self.assertEqual(result["severity"], "high")

    def test_code_fence_stripped(self):
        fenced = "```json\n" + json.dumps(make_valid_response()) + "\n```"
        with patch("agent._client") as mock_client:
            mock_create = MagicMock()
            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content=fenced))
            ]
            mock_client.return_value.chat.completions.create = mock_create
            with patch.dict(os.environ, {"NEBIUS_API_KEY": "v1.test"}):
                result = triage("XSS in comment field")
            self._assert_schema(result)

    def test_missing_api_key_exits(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                triage("test")

    def test_system_prompt_enforces_strict_json(self):
        self.assertIn("strict json", SYSTEM_PROMPT.lower())
        self.assertIn("No markdown", SYSTEM_PROMPT)

    def test_user_template_contains_placeholder(self):
        self.assertIn("{report}", USER_TEMPLATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
