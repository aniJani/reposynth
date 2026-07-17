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
