import json
tasks = json.load(open(r"\data\frc\exports\snapshot_260156.json"))
print("total:", len(tasks))
# find a task that HAS annotations and show its shape
for t in tasks:
    anns = t.get("annotations") or t.get("completions") or []
    if anns:
        import pprint
        print("KEYS:", list(t.keys()))
        print("ANN KEYS:", list(anns[0].keys()))
        print("RESULT SAMPLE:")
        pprint.pprint(anns[0].get("result", [])[:2])
        break
else:
    print("no task with annotations found under 'annotations' or 'completions'")

# also: how many have each field?
has_ann = sum(1 for t in tasks if t.get("annotations"))
has_comp = sum(1 for t in tasks if t.get("completions"))
print("has annotations:", has_ann, " has completions:", has_comp)
