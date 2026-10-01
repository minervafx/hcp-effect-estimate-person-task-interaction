"""Configurable file locations.

No HCP data ship with this repository. Every location below is under a work directory
chosen by the user (HCPEE_WORK, default ./work), and each can be overridden
individually with the environment variable named next to it.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.abspath(os.environ.get("HCPEE_WORK", os.path.join(REPO, "work")))


def _p(env, default):
    return os.path.abspath(os.environ.get(env, default))


INPUTS = os.path.join(REPO, "inputs")
SUBJECTS_FILE = _p("HCPEE_SUBJECTS", os.path.join(INPUTS, "included_subjects.txt"))
# Group HCP-MMP1.0 dlabel (user-supplied, see README) and the 379-label vector built from it
ATLAS_DLABEL = _p("HCPEE_ATLAS_DLABEL", os.path.join(WORK, "atlas", "HCP_MMP1_group.dlabel.nii"))
ATLAS_LABELS = _p("HCPEE_ATLAS_LABELS", os.path.join(WORK, "atlas", "labels379.npy"))
# Parcellated COPE (effect-estimate) cache written by scripts/01_fetch_effect_estimates.py
EE_CACHE = _p("HCPEE_EE_CACHE", os.path.join(WORK, "effect_estimate_cache"))
# Parcellated resting-state timeseries written by scripts/02_fetch_rest.py
REST_CACHE = _p("HCPEE_REST_CACHE", os.path.join(WORK, "rest_cache"))
# FreeSurfer-derived anatomy blocks written by scripts/03_fetch_anatomy.py
ANAT_DIR = _p("HCPEE_ANAT_DIR", os.path.join(WORK, "anatomy"))
OUT = _p("HCPEE_OUT", os.path.join(WORK, "outputs"))
# Optional: directory holding the two frozen analysis plans, used only to record their
# SHA-256 in the outputs. If absent, outputs record null plus the expected hash.
PREREG_DIR = _p("HCPEE_PREREG_DIR", os.path.join(REPO, "preregistration"))
PREREG_EXPECTED_SHA256 = {
    "HCP_EFFECT_ESTIMATE_FALSIFICATION_PREREG.md":
        "841936f5858a172ef5f2ee834116313eb8abfc6e8304969647dfa667dd3aed5e",
    "HCP_EFFECT_ESTIMATE_CONTROLS_PREREG.md":
        "39586d06a96948b428187c69f8b14cf702fe02830852160080f5864ef152fd24",
}


def prereg_sha(name):
    """SHA-256 of a frozen plan if present locally, else None (expected hash is recorded
    separately by the caller)."""
    import hashlib
    p = os.path.join(PREREG_DIR, name)
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
