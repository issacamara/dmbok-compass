"""Keep a newly provisioned parser on the image currently serving the API."""

import os
import re
import subprocess


def output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    project = os.environ["PROJECT_ID"]
    scope = (f"--project={project}", "--region=europe-west1", "--quiet")

    def ready_image() -> str:
        revision = output("gcloud", "run", "services", "describe", "dmbok-compass-api", *scope,
                          "--format=value(status.latestReadyRevisionName)")
        if not revision:
            raise RuntimeError("API has no ready revision; retry parser reconciliation after deployment")
        return output("gcloud", "run", "revisions", "describe", revision, *scope,
                      "--format=value(spec.containers[0].image)")

    for _ in range(3):
        image = ready_image()
        if image == "us-docker.pkg.dev/cloudrun/container/hello":
            print("Initial API placeholder: parser image will be set by the first application deployment")
            return
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
            raise RuntimeError("Ready API image is not digest-pinned; retry after an immutable deployment")
        subprocess.run(("gcloud", "run", "jobs", "update", "dmbok-compass-parser", *scope,
                        f"--image={image}"), check=True)
        if ready_image() == image:
            return
    raise RuntimeError("API image changed during reconciliation; retry the infrastructure workflow")


if __name__ == "__main__":
    main()
