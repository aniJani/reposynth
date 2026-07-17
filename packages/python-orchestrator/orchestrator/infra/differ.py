"""Structural drift diff between two StateDocs."""
from .state_doc import section_hash

# section name -> (list field, identity key)
_KEYED = {"schema": ("tables", "name"), "rls": ("tables", "table"),
          "storage": ("buckets", "name"), "functions": ("list", "name"),
          "collections": ("list", "collectionId")}


def _keyed_diff(pa: dict, pb: dict, list_field: str, key: str) -> dict:
    items_a = {i[key]: i for i in pa.get(list_field, [])}
    items_b = {i[key]: i for i in pb.get(list_field, [])}
    added = sorted(set(items_b) - set(items_a))
    removed = sorted(set(items_a) - set(items_b))
    changed = sorted(k for k in set(items_a) & set(items_b)
                     if section_hash(items_a[k]) != section_hash(items_b[k]))
    return {"added": added, "removed": removed, "changed": changed}


def diff(a: dict, b: dict) -> dict:
    sa, sb = a.get("sections", {}), b.get("sections", {})
    sections, unchanged = {}, []
    for name in sorted(set(sa) | set(sb)):
        if name not in sa:
            sections[name] = {"status": "added"}
            continue
        if name not in sb:
            sections[name] = {"status": "removed"}
            continue
        if sa[name].get("hash") == sb[name].get("hash"):
            unchanged.append(name)
            continue
        if name in _KEYED:
            sections[name] = _keyed_diff(sa[name], sb[name], *_KEYED[name])
        elif name == "auth":
            pa, pb = set(sa[name].get("providers", [])), set(sb[name].get("providers", []))
            sections[name] = {"added": sorted(pb - pa), "removed": sorted(pa - pb),
                              "changed": ["settings"] if sa[name].get("settings") != sb[name].get("settings") else []}
        elif name == "config":
            na, nb = set(sa[name].get("envNames", [])), set(sb[name].get("envNames", []))
            ha, hb = sa[name].get("valueHashes", {}), sb[name].get("valueHashes", {})
            sections[name] = {"added": sorted(nb - na), "removed": sorted(na - nb),
                              "rotated": sorted(k for k in na & nb if ha.get(k) != hb.get(k))}
        elif name == "rules":
            ha = {s["service"]: s.get("contentSha256") for s in sa[name].get("services", [])}
            hb = {s["service"]: s.get("contentSha256") for s in sb[name].get("services", [])}
            sections[name] = {"added": sorted(set(hb) - set(ha)),
                              "removed": sorted(set(ha) - set(hb)),
                              "changed": sorted(k for k in set(ha) & set(hb) if ha[k] != hb[k])}
        else:
            sections[name] = {"changed": ["<section content>"]}
    return {"sections": sections, "unchanged": unchanged}
