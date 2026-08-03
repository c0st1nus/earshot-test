import sys
import json

def main():
    if len(sys.argv) != 3:
        print("Usage: audacity_labels_to_json.py <labels.txt> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    speech = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                continue

            start_sec = float(parts[0])
            end_sec = float(parts[1])
            label = parts[2].strip().lower()

            if label in {"speech", "voice", "voiced", "s"}:
                speech.append(
                    {
                        "start_ms": int(start_sec * 1000),
                        "end_ms": int(end_sec * 1000),
                    }
                )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"speech": speech}, f, indent=2)

    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
