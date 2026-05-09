#!/usr/bin/env python3
"""Pet plugin tests — moved from cortex-core/src/test_cortex.py in slice 2c2a.

Runs without BLE, display, or any hardware dependencies. Uses the same
plain `check()` style as the core test_cortex.py — kept consistent so a
single test invocation can run both side by side.

Usage:
    python plugins/pet/tests/test_pet.py
"""

import json
import os
import sys

# Wire up sys.path so this file can run from anywhere.
# Layout: cortex-core/plugins/pet/tests/test_pet.py
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_TESTS_DIR)                              # plugins/pet
_REPO_ROOT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))             # cortex-core
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))                    # cortex_db, cortex_protocol
sys.path.insert(0, _PLUGIN_DIR)                                        # pet, body_shell, etc.

from cortex_db import CortexDB
from cortex_protocol import CortexProtocol

DB_PATH = "/tmp/test_pet_plugin.db"
PASSED = 0
FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name}")


def test_pet_db():
    print("\n=== Pet DB Tests ===")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = CortexDB(DB_PATH)

    # Pet state
    db.set_pet_state("stage", "0")
    val = db.get_pet_state("stage")
    check("set/get pet_state", val == "0")

    # Update pet state (upsert)
    db.set_pet_state("stage", "2")
    val = db.get_pet_state("stage")
    check("pet_state upsert", val == "2")

    # Get non-existent key
    val = db.get_pet_state("nonexistent", default="fallback")
    check("pet_state default", val == "fallback")

    # Get all pet state
    db.set_pet_state("mood", "happy")
    all_state = db.get_all_pet_state()
    check("get_all_pet_state", all_state.get("stage") == "2")
    check("get_all_pet_state mood", all_state.get("mood") == "happy")

    # Insert pet interaction
    pid = db.insert_pet_interaction(
        prompt="hello pet",
        response="hi there!",
        sentiment_score=0.5,
        inference_time_ms=1200,
        tokens_generated=10,
        stage=1,
        mood="happy",
    )
    check("insert_pet_interaction returns id", pid >= 1)

    # Insert more interactions for mood scoring
    for i in range(5):
        db.insert_pet_interaction(
            prompt="test {}".format(i),
            response="resp",
            sentiment_score=0.8,  # positive
        )
    for i in range(5):
        db.insert_pet_interaction(
            prompt="bad {}".format(i),
            response="resp",
            sentiment_score=-0.5,  # negative
        )

    # Mood score (weighted average of recent)
    mood = db.get_pet_mood_score(window=20)
    check("pet_mood_score is float", isinstance(mood, float))
    check("pet_mood_score in range", -1.0 <= mood <= 1.0)

    # Interaction count
    count = db.get_pet_interaction_count()
    check("pet_interaction_count", count == 11)

    # Recent interactions
    recent = db.get_recent_pet_interactions(5)
    check("get_recent_pet_interactions limit", len(recent) == 5)
    check("recent ordered desc", recent[0]["id"] > recent[-1]["id"])

    # Update interaction
    db.update_pet_interaction(pid, "updated response", 2000, 20)
    updated = db.get_recent_pet_interactions(20)
    found = [r for r in updated if r["id"] == pid]
    check("update_pet_interaction", found[0]["response"] == "updated response")
    check("update_pet_interaction time", found[0]["inference_time_ms"] == 2000)

    # Pet stats
    stats = db.get_pet_stats()
    check("pet_stats has total_interactions", stats["total_interactions"] == 11)
    check("pet_stats has mood_score", "mood_score" in stats)
    check("pet_stats has state", isinstance(stats["state"], dict))

    # Context includes pet
    ctx = db.get_context()
    check("context has pet", "pet" in ctx)
    check("context pet has total_interactions", "total_interactions" in ctx["pet"])

    db.close()
    os.remove(DB_PATH)
    print("  Pet DB tests complete.")


def test_pet_protocol():
    print("\n=== Pet Protocol Tests ===")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = CortexDB(DB_PATH)

    # Test protocol WITHOUT pet engine (pet=None)
    proto_no_pet = CortexProtocol(db, pet=None)
    resp = proto_no_pet.handle_message('CMD:pet_ask:{"prompt":"hello"}')
    check("pet_ask no engine -> ERR", resp.startswith("ERR:pet_ask:"))

    resp = proto_no_pet.handle_message("CMD:pet_status")
    check("pet_status no engine -> ERR", resp.startswith("ERR:pet_status:"))

    resp = proto_no_pet.handle_message("CMD:pet_mood")
    check("pet_mood no engine -> ERR", resp.startswith("ERR:pet_mood:"))

    resp = proto_no_pet.handle_message("CMD:pet_response")
    check("pet_response no engine -> ERR", resp.startswith("ERR:pet_response:"))

    # pet_ask missing prompt
    resp = proto_no_pet.handle_message("CMD:pet_ask:{}")
    check("pet_ask missing prompt -> ERR", resp.startswith("ERR:pet_ask:"))

    # pet_history (uses DB directly, should work without engine)
    proto_with_db = CortexProtocol(db, pet=None)
    # Insert some test interactions
    db.insert_pet_interaction(prompt="test", response="response", mood="happy")
    resp = proto_with_db.handle_message("CMD:pet_history:{}")
    check("pet_history -> RSP", resp.startswith("RSP:pet_history:"))
    history = json.loads(resp[16:])
    check("pet_history returns list", isinstance(history, list))
    check("pet_history has data", len(history) >= 1)

    # Query pet tables
    resp = proto_with_db.handle_message(
        'CMD:query:{"table":"pet_interactions","limit":5}'
    )
    check("query pet_interactions -> RSP", resp.startswith("RSP:query:"))

    resp = proto_with_db.handle_message(
        'CMD:query:{"table":"pet_state","limit":5}'
    )
    check("query pet_state -> RSP", resp.startswith("RSP:query:"))

    db.close()
    os.remove(DB_PATH)
    print("  Pet protocol tests complete.")


def test_pet_sentiment():
    print("\n=== Pet Sentiment Tests ===")
    from pet import simple_sentiment

    # Positive
    score = simple_sentiment("thank you so much you are amazing")
    check("positive sentiment > 0", score > 0)

    # Negative
    score = simple_sentiment("you are stupid and boring")
    check("negative sentiment < 0", score < 0)

    # Neutral
    score = simple_sentiment("the weather today")
    check("neutral sentiment == 0", score == 0.0)

    # Mixed — both have similar weights, should be near zero
    score = simple_sentiment("good but also bad")
    check("mixed sentiment near 0", abs(score) < 0.5)

    # Negation handling
    score = simple_sentiment("not good")
    check("negation 'not good' < 0", score < 0)

    score = simple_sentiment("not bad")
    check("negation 'not bad' > 0", score > 0)

    # Intensity modifiers
    score_plain = simple_sentiment("good")
    score_very = simple_sentiment("very good")
    check("'very good' > 'good'", score_very > score_plain)

    score_extreme = simple_sentiment("extremely amazing")
    check("'extremely amazing' strongly positive", score_extreme > 0.8)

    # Strong positive
    score = simple_sentiment("I love this, it's perfect and amazing!")
    check("strong positive > 0.7", score > 0.7)

    # Strong negative
    score = simple_sentiment("this is horrible and disgusting")
    check("strong negative < -0.5", score < -0.5)

    # Punctuation boost
    score_no_excl = simple_sentiment("that is great")
    score_excl = simple_sentiment("that is great!")
    check("exclamation boosts magnitude", abs(score_excl) >= abs(score_no_excl))

    # Empty and edge cases
    check("empty string -> 0", simple_sentiment("") == 0.0)
    check("just punctuation", simple_sentiment("!!!") == 0.1)  # excl heuristic
    check("unknown words -> 0", simple_sentiment("xyzzy foobar quux") == 0.0)

    # Greetings are mildly positive
    score = simple_sentiment("hello! how are you?")
    check("greeting mildly positive", score > 0)

    print("  Pet sentiment tests complete.")


def test_pet_engine_stages():
    print("\n=== Pet Engine Stage Tests ===")
    from pet import PetEngine

    # Test stage calculation (static method)
    check("stage 0 at count 0", PetEngine._count_to_stage(0) == 0)
    check("stage 0 at count 49", PetEngine._count_to_stage(49) == 0)
    check("stage 1 at count 50", PetEngine._count_to_stage(50) == 1)
    check("stage 1 at count 199", PetEngine._count_to_stage(199) == 1)
    check("stage 2 at count 200", PetEngine._count_to_stage(200) == 2)
    check("stage 3 at count 1000", PetEngine._count_to_stage(1000) == 3)
    check("stage 4 at count 5000", PetEngine._count_to_stage(5000) == 4)
    check("stage 4 at count 99999", PetEngine._count_to_stage(99999) == 4)

    # Test mood mapping (static method)
    check("mood happy at score 0.5", PetEngine._score_to_mood(0.5) == "happy")
    check("mood content at score 0.2", PetEngine._score_to_mood(0.2) == "content")
    check("mood neutral at score 0.0", PetEngine._score_to_mood(0.0) == "neutral")
    check("mood uneasy at score -0.2", PetEngine._score_to_mood(-0.2) == "uneasy")
    check("mood sad at score -0.5", PetEngine._score_to_mood(-0.5) == "sad")

    print("  Pet engine stage tests complete.")


def test_pet_analytics():
    print("\n=== Pet Analytics Tests ===")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = CortexDB(DB_PATH)

    # Insert test interactions across different moods
    for i in range(10):
        db.insert_pet_interaction(
            prompt="happy test {}".format(i),
            response="response",
            sentiment_score=0.6,
            inference_time_ms=1500 + i * 100,
            tokens_generated=10 + i,
            stage=1,
            mood="happy",
        )
    for i in range(5):
        db.insert_pet_interaction(
            prompt="sad test {}".format(i),
            response="response",
            sentiment_score=-0.4,
            inference_time_ms=800,
            tokens_generated=5,
            stage=1,
            mood="sad",
        )

    # Test DB analytics method
    analytics = db.get_pet_analytics(days=7)
    check("analytics has period_days", analytics["period_days"] == 7)
    check("analytics has total_interactions", analytics["total_interactions"] == 15)
    check("analytics has daily_trend", isinstance(analytics["daily_trend"], list))
    check("analytics has mood_distribution", isinstance(analytics["mood_distribution"], list))
    check("analytics has performance", isinstance(analytics["performance"], dict))
    check("analytics has stage_progression", isinstance(analytics["stage_progression"], list))

    # Mood distribution should have happy and sad
    moods = {d["mood"]: d["count"] for d in analytics["mood_distribution"]}
    check("analytics mood dist happy", moods.get("happy") == 10)
    check("analytics mood dist sad", moods.get("sad") == 5)

    # Performance stats
    perf = analytics["performance"]
    check("analytics avg_ms exists", perf.get("avg_ms") is not None)

    # Test via protocol
    proto = CortexProtocol(db)
    resp = proto.handle_message('CMD:pet_analytics:{"days":7}')
    check("CMD:pet_analytics -> RSP", resp.startswith("RSP:pet_analytics:"))
    result = json.loads(resp[18:])
    check("pet_analytics has total_interactions", "total_interactions" in result)

    # Test with 0 days (edge case)
    resp = proto.handle_message('CMD:pet_analytics:{"days":0}')
    check("pet_analytics 0 days -> RSP", resp.startswith("RSP:pet_analytics:"))

    db.close()
    os.remove(DB_PATH)
    print("  Pet analytics tests complete.")


def main():
    global PASSED, FAILED
    print("Pet Plugin — Test Suite")
    print("=" * 40)

    test_pet_db()
    test_pet_protocol()
    test_pet_sentiment()
    test_pet_engine_stages()
    test_pet_analytics()

    print("\n" + "=" * 40)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED > 0:
        sys.exit(1)
    else:
        print("All pet tests passed!")


if __name__ == "__main__":
    main()
