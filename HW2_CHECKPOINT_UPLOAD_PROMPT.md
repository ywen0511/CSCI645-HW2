# Deliver the Existing HW2 Checkpoints

## Task and authorization

Repository: **`ywen0511/CSCI645-HW2`**  
Submission branch: **`submission/wen-yan-hw2`**  
Release tag: **`hw2-v2-checkpoints`**

The revised experiments have been reviewed. Your task is to finish delivery of the existing checkpoint artifacts. **No additional training, demonstration collection, or evaluation is requested.**

You are explicitly authorized to create the GitHub release named above in this repository and upload the existing checkpoint bundles and their checksum file. Keep the repository's current visibility unchanged. Do not ask again for approval to perform this specific upload, unless an applicable platform restriction requires it.

Use the repository, its artifact records, and artifacts available in your existing execution environment. This task does not depend on any previous chat or files on the user's computer.

## 1. Locate and verify the existing bundles

Locate these artifacts from the completed revision:

- `hw2_ckpt_act_quantity.tar`
- `hw2_ckpt_act_noise.tar`
- `hw2_ckpt_flow_and_teacher.tar`

Read `COMPLETION_REPORT.md` and `WEN_YAN_HW2/results/checkpoints_manifest.json` for their provenance and expected hashes. Check their actual SHA-256 values against the manifest.

Inspect each archive without executing its contents. Verify that the bundles collectively contain:

- All 21 retrained student checkpoints: seven conditions, each with training seeds 0, 1, and 2.
- The retrained PPO teacher checkpoint.
- The identities, configurations, normalization information, and other metadata required by the current loading and evaluation scripts.

Check each available checkpoint against its corresponding manifest entry. Verify the archive layout and establish the exact extraction destination expected by the scripts, including how it relates to `HW2_RUNS`.

If a bundle is missing but its original constituent files are available, you may recreate that bundle without changing the model files. Record any changed archive hash and its cause. Do not modify an expected checksum merely to hide an unexplained mismatch.

If required checkpoint files are missing or have unexplained mismatches, report exactly which artifacts are affected. Do not silently retrain, substitute another model, or describe an incomplete set as complete.

## 2. Publish to GitHub

1. Check whether `hw2-v2-checkpoints` already exists. If it does, inspect its target commit and assets before making changes. Do not delete or overwrite unrelated release contents or move an existing tag without approval.
2. If a new release is needed, identify the existing submission commit matching these artifacts from the repository's provenance records. Create the release against that verified commit, rather than assuming the current branch tip has the same artifact versions.
3. Upload the three verified checkpoint bundles and a `SHA256SUMS.txt` file. Include the checkpoint manifest as an asset if it helps make the downloads self-contained.
4. In the release notes, identify the corresponding source commit and explain that these are the retrained protocol-v2 checkpoints. Distinguish the retrained teacher from the original teacher that collected the committed demonstrations.
5. Preserve the repository's current visibility. Use this repository for delivery; do not publish the artifacts through an unrelated service.

If an identical verified asset is already present, reuse it. If an asset name conflicts with different contents, investigate and report the conflict rather than replacing it blindly.

## 3. Verify delivery

Verify the actual uploaded assets by downloading them through the release's download mechanism and recomputing their SHA-256 checksums. Listing an asset name or receiving an upload-success response alone is not sufficient.

Check that the downloaded archives are readable and contain the intended files. Confirm that their extraction layout matches the paths required by the evaluation scripts. Do not run a full evaluation for this delivery task.

If the repository is private, explain any authentication needed to download the assets; do not claim anonymous access. Never include tokens, credentials, or private keys in release notes or commands.

## 4. Return a concise handoff

Return all of the following:

- Release URL and target commit URL.
- Direct links to each checkpoint bundle, checksum file, and any manifest asset.
- A table of filenames, sizes, SHA-256 hashes, and verification status.
- Exact commands to download, verify, and extract the bundles into the expected `HW2_RUNS` directory.
- A description of the expected directory layout after extraction.
- A clear statement of whether all 21 student checkpoints and the retrained teacher were delivered successfully.
- Any missing artifacts, permission failures, or unresolved inconsistencies.

Make the commands usable from a fresh checkout. Use configurable paths and repository-relative references, not historical machine-specific directories.

## Scope boundaries

- Do not retrain models, collect demonstrations, rerun experiments, or alter experimental results.
- Do not edit the homework report, its source, figures, tables, or reproduction README in this task. Those final corrections are being handled separately; return the verified links and commands for inclusion there.
- Do not upload or publish the final homework submission ZIP yet. It must be built after the report and documentation corrections are complete.
- Do not merge branches, force-push, change repository visibility, purchase services, or submit anything to Brightspace.
- Do not stop at a plan or an attempted upload. Complete and verify delivery, or report a concrete blocker and the exact incomplete items.
