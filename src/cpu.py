from pathlib import Path

possible = [
    Path(r"C:/Users/Dragroyale/Desktop"),
    Path(r"C:/Users/Dragroyale/Downloads"),
    Path(r"C:/Users/Dragroyale/Documents"),
    Path(r"C:/Users/Dragroyale/Desktop/content"),
]

for base in possible:
    print(f"\nSearching in: {base}")
    if base.exists():
        for p in base.rglob("*.csv"):
            if "WorkingHours" in p.name or "pcap_ISCX" in p.name:
                print(p)