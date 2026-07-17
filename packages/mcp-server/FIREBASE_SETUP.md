# Firebase connector setup (read-only)

The Firebase connector reads live project state (Firestore collections/indexes,
security rules, auth, storage, functions). It is read-only by construction and
**refuses to run if its credential holds write permissions** (a fail-closed IAM probe).

## 1. Create a dedicated read-only service account
Do NOT reuse the default App Engine/Compute service account — it may carry legacy
Editor access. Create a fresh one and bind only:
- `roles/firebase.viewer`
- `roles/datastore.viewer`
- `roles/firebaserules.viewer`
- a **custom role** with: `storage.buckets.list`, `storage.buckets.get`,
  `storage.buckets.getIamPolicy`, `storage.objects.list`
  (no predefined role isolates bucket enumeration without write access)

## 2. Enable the APIs
firestore, firebaserules, identitytoolkit, cloudfunctions, storage,
firebasestorage, cloudresourcemanager.

## 3. Configure the target
`.reposynth/targets.json`:
```jsonc
{ "targets": { "prod-fb": {
    "connector": "firebase", "projectId": "my-app",
    "databaseId": "(default)", "credentialsEnv": "FIREBASE_SA_JSON_PATH",
    "risk": "prod" } } }
```
Set `FIREBASE_SA_JSON_PATH` to the path of the service-account JSON key.

## 4. Verify scoping
Run any `infra_*` tool once. If the credential can write, the probe refuses with a
clear error listing the offending permissions — fix the role bindings and retry.

## Scope (v1)
Firestore **Native mode only** (Datastore mode errors; Realtime Database unsupported).
Indexes are capture-only (no `index_exists`); collection checks use `collection_exists`
(note: an empty Firestore collection reports absent). `env_name_present` is unsupported.
