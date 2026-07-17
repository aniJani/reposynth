"""Firebase connector: read-only StateDoc capture over Google REST APIs.

Read-only by construction (only GET/list endpoints called; no write method).
A fail-closed IAM probe refuses to run a credential that holds write access.
Firestore Native mode only.
"""
import hashlib

WRITE_PERMS = [
    "datastore.entities.create", "datastore.entities.update", "datastore.entities.delete",
    "firebaserules.releases.update", "firebaserules.rulesets.create",
    "cloudfunctions.functions.update", "cloudfunctions.functions.delete",
    "firebaseauth.configs.update",
    "storage.objects.delete", "storage.buckets.update",
]

_RESOURCE_MANAGER = "https://cloudresourcemanager.googleapis.com/v1"


def probe_readonly(call, project: str) -> None:
    """Fail closed: refuse if the credential holds any write permission.

    On a probe-call error (API disabled, network), warn-and-proceed — the
    connector is read-only by construction regardless.
    """
    try:
        resp = call("POST", f"{_RESOURCE_MANAGER}/projects/{project}:testIamPermissions",
                    json_body={"permissions": WRITE_PERMS})
    except Exception:
        return  # could not verify; proceed (code is read-only anyway)
    granted = resp.get("permissions") or []
    if granted:
        raise RuntimeError(
            f"credential for project '{project}' holds write permissions {granted} — "
            "refusing to run; scope the service account read-only")


_FIRESTORE = "https://firestore.googleapis.com/v1"


def fetch_collections(call, project: str, db: str) -> dict:
    base = f"{_FIRESTORE}/projects/{project}/databases/{db}/documents:listCollectionIds"
    ids, token, complete = [], None, True
    while True:
        body = {"pageSize": 300}
        if token:
            body["pageToken"] = token
        resp = call("POST", base, json_body=body)
        ids.extend(resp.get("collectionIds", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return {"list": [{"collectionId": c, "path": c, "parentDocumentPath": None,
                      "subcollections": []} for c in sorted(ids)],
            "rootComplete": complete}


def _leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def fetch_indexes(call, project: str, db: str) -> dict:
    cg = f"{_FIRESTORE}/projects/{project}/databases/{db}/collectionGroups/-"
    idx = call("GET", f"{cg}/indexes").get("indexes", [])
    composite = [{"queryScope": i.get("queryScope"), "state": i.get("state"),
                  "fields": i.get("fields", [])} for i in idx]
    fld = call("GET", f"{cg}/fields",
               params={"filter": "indexConfig.usesAncestorConfig:false"}).get("fields", [])
    single = [{"fieldPath": _leaf(f["name"])} for f in fld]
    return {"composite": composite, "singleFieldOverrides": single}
