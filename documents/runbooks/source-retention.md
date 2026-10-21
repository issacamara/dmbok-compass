# Source-generation retention runbook

Firestore backup retention and corpus-source retention are separate controls.
`infra/terraform/backup.tf` retains at most 30 daily Firestore backup copies;
that limit does not authorize deletion of versioned source objects.

## Safe inventory

Run `scripts/source_retention.py` from the reviewed application image with an
authenticated runtime principal, the approved image digest, and a reviewed
snapshot of release references and source metadata:

```sh
python scripts/source_retention.py snapshot.json \
  --actor SERVICE_ACCOUNT \
  --image-digest sha256:APPROVED \
  --expected-image-digest sha256:APPROVED
```

The inventory protects the exact bucket, object, generation, and content hash
of the active release and one validated rollback release. Explicitly retained
generations are protected too. An old, unreferenced source is eligible only
when its exact identity has a verified rebuild/recovery path; otherwise it is
reported as refused. A changed release-reference fingerprint aborts the
inventory and requires a fresh snapshot.

This ticket intentionally has no delete operation and no Terraform apply. Any
future deletion must revalidate release references, verify the same recovery
path, record the approved retention policy and actor, and use a separately
reviewed operation.
