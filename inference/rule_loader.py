import json

def load_rules(filepaths):
    rules = []
    for path in filepaths:
        with open(path, encoding='utf-8') as f:
            rules.extend(json.load(f))
    return rules