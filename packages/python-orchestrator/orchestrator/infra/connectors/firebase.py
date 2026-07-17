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


_RULES = "https://firebaserules.googleapis.com/v1"


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_rules(call, project: str) -> dict:
    releases, token = [], None
    while True:
        params = {"pageToken": token} if token else None
        resp = call("GET", f"{_RULES}/projects/{project}/releases", params=params)
        releases.extend(resp.get("releases", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    services = []
    for rel in releases:
        rel_leaf = rel["name"].split("/releases/", 1)[-1]  # e.g. cloud.firestore | firebase.storage/bkt
        if rel_leaf.startswith("cloud.firestore"):
            service, scope = "cloud.firestore", rel_leaf[len("cloud.firestore"):].lstrip("/") or "(default)"
        elif rel_leaf.startswith("firebase.storage"):
            service, scope = "firebase.storage", rel_leaf[len("firebase.storage"):].lstrip("/")
        else:
            continue
        ruleset = call("GET", f"{_RULES}/{rel['rulesetName']}")
        content = "\n".join(f.get("content", "") for f in ruleset.get("source", {}).get("files", []))
        services.append({"service": service, "scope": scope,
                         "releaseName": rel["name"], "rulesetName": rel["rulesetName"],
                         "content": content, "contentSha256": _sha256(content)})
    return {"services": services}


_IDENTITY = "https://identitytoolkit.googleapis.com/admin/v2"


def fetch_auth(call, project: str) -> dict:
    cfg = call("GET", f"{_IDENTITY}/projects/{project}/config")
    signin = cfg.get("signIn", {})
    providers = []
    if signin.get("email", {}).get("enabled"):
        providers.append("password")
    if signin.get("phoneNumber", {}).get("enabled"):
        providers.append("phone")
    if signin.get("anonymous", {}).get("enabled"):
        providers.append("anonymous")
    idps = call("GET", f"{_IDENTITY}/projects/{project}/defaultSupportedIdpConfigs")
    for idp in idps.get("defaultSupportedIdpConfigs", []):
        if idp.get("enabled"):
            providers.append(idp["name"].rsplit("/", 1)[-1])
    return {"providers": sorted(providers),
            "settings": {"tier": cfg.get("subtype"),
                         "authorizedDomains": cfg.get("authorizedDomains", []),
                         "mfaState": cfg.get("mfa", {}).get("state")}}


_FB_STORAGE = "https://firebasestorage.googleapis.com/v1beta"
_GCS = "https://storage.googleapis.com/storage/v1"
_PUBLIC_MEMBERS = {"allUsers", "allAuthenticatedUsers"}


def fetch_storage(call, project: str) -> dict:
    listed = call("GET", f"{_FB_STORAGE}/projects/{project}/buckets").get("buckets", [])
    out = []
    for entry in listed:
        name = _leaf(entry.get("name", ""))
        meta = call("GET", f"{_GCS}/b/{name}")
        pap = meta.get("iamConfiguration", {}).get("publicAccessPrevention")
        iam = call("GET", f"{_GCS}/b/{name}/iam")
        public_via_iam = any(m in _PUBLIC_MEMBERS
                             for binding in iam.get("bindings", [])
                             for m in binding.get("members", []))
        bucket = {"name": name, "publicViaIAM": public_via_iam,
                  "publicViaSecurityRules": None}
        if pap == "enforced":
            bucket["public"] = False
        elif public_via_iam:
            bucket["public"] = True
        # else: private-via-IAM + rules unknown -> omit `public` (honest unknown)
        out.append(bucket)
    return {"buckets": out}


_FUNCTIONS = "https://cloudfunctions.googleapis.com/v2"


def fetch_functions(call, project: str) -> dict:
    resp = call("GET", f"{_FUNCTIONS}/projects/{project}/locations/-/functions")
    out = []
    for f in resp.get("functions", []):
        parts = f["name"].split("/")
        region = parts[parts.index("locations") + 1] if "locations" in parts else None
        gen = "gen2" if f.get("environment") == "GEN_2" else "gen1"
        trigger = "https" if f.get("serviceConfig", {}).get("uri") or f.get("httpsTrigger") \
            else ("event" if f.get("eventTrigger") else None)
        out.append({"name": _leaf(f["name"]), "status": f.get("state") or f.get("status"),
                    "region": region, "generation": gen, "triggerType": trigger})
    return {"list": out, "unreachable": resp.get("unreachable", [])}
