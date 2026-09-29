import json
import sys

from car_steer.cli import main


def test_local_cli_jsonl_round_trip(tiny_model, tokenizer, tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    tiny_model.save_pretrained(weights)
    tokenizer.save_pretrained(weights)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "example", "context": "Mira Sen is the director.",
                                    "question": "Who is the director?"}) + "\n", encoding="utf-8")
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"threshold": 2., "max_new_tokens": 3}), encoding="utf-8")
    output = tmp_path / "answers.jsonl"
    monkeypatch.setattr(sys, "argv", ["car-steer", "--model", str(weights), "--config", str(config),
                                     "--input", str(questions), "--output", str(output),
                                     "--device", "cpu", "--local-files-only"])
    main()
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["id"] == "example" and 1 <= len(record["token_ids"]) <= 3
    assert record["prefill"]["trigger_center"] is None
    assert record["text"] == tokenizer.decode(record["token_ids"], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
