# Differences from the code that produced the manuscript's saved outputs

Every shipped file is listed with its project origin. Files marked **verbatim** are byte-identical to the originals. All others are shown as unified diffs. The changes are path/import wiring, the map-type guard, and removal of unused code (twin estimators, EEG loaders, and the z-statistic task-cache reader). No estimator, seed, permutation count or parameter was changed. `docs/REPRODUCIBILITY.md` documents that the patched code regenerates the saved outputs.

Removed runs of three or more lines are collapsed into a one-line marker, and two original tokens are redacted: a local credential-file path and an internal user-agent label. The original files' SHA-256 values are listed for exact verification by the author.

| Shipped file | Origin (project `src/`) | Status | Original SHA-256 |
|---|---|---|---|
| `hcpee/cifti.py` | `phase2_9/cifti.py` | verbatim | `08c4ba10a92b3318…` |
| `hcpee/hcp_s3.py` | `phase2_9/hcp_s3.py` | patched | `253e8fb70bd29d11…` |
| `hcpee/s3zip.py` | `phase2_9/s3zip.py` | verbatim | `2b2cd1945b3f2c78…` |
| `hcpee/p29.py` | `phase2_9/p29.py` | patched | `1b24694045e884dc…` |
| `hcpee/contrast.py` | `phase2_5/contrast.py` | patched | `5ddfbe2f3077a838…` |
| `hcpee/analysis.py` | `phase2/analysis.py` | verbatim | `4ffda831a34cf625…` |
| `hcpee/p30.py` | `phase3_0/p30.py` | patched | `eae0301373c06ab1…` |
| `hcpee/ix.py` | `phase3_1/ix.py` | verbatim | `a5434dd3ea84644c…` |
| `hcpee/an.py` | `phase3_2/an.py` | verbatim | `4789f22a074c1c0d…` |
| `scripts/01_fetch_effect_estimates.py` | `phase2_9/02_fetch_effect_estimate.py` | patched | `f420049b377964bc…` |
| `scripts/02_fetch_rest.py` | `phase3_0/01_fetch_rest.py` | patched | `dad79450bb3116f0…` |
| `scripts/10_interaction.py` | `phase3_1/01_interaction_effect_estimate.py` | patched | `62bc1672af1acefa…` |
| `scripts/11_rest_control.py` | `phase3_1/02_rest_control_effect_estimate.py` | patched | `46b1c790c6823228…` |
| `scripts/12_anatomy_control.py` | `phase3_1/01_anatomy_interaction_effect_estimate.py` | patched | `2df2e9a79d50c464…` |
| `scripts/03_fetch_anatomy.py` | functions copied verbatim from `phase2_9/02_fetch.py` | new wrapper | — |
| `scripts/00_build_atlas.py` | new; reproduces the historical `labels379.npy` exactly | new | — |
| `scripts/20_verify_expected.py`, `scripts/30_make_figures.py` (drawing code from the manuscript figure script) | new/adapted | new | — |
| `hcpee/paths.py`, `hcpee/map_type_guard.py`, `tests/test_map_type_guard.py` | new | new | — |

## `hcpee/hcp_s3.py`  (from `phase2_9/hcp_s3.py`)

```diff
--- original/phase2_9/hcp_s3.py
+++ release/hcpee/hcp_s3.py
@@ -15,3 +15,3 @@
 
-    <local credential path>
+    ~/.hcp_aws.env      (or the path in $HCP_CRED_FILE)
         HCP_AWS_ACCESS_KEY_ID=...
@@ -29,3 +29,3 @@
 
-CRED_FILE = os.environ.get("HCP_CRED_FILE", "<local credential path>")
+CRED_FILE = os.environ.get("HCP_CRED_FILE", os.path.expanduser("~/.hcp_aws.env"))
 BUCKET = "hcp-openaccess"
@@ -33,3 +33,3 @@
 SERVICE = "s3"
-UA = "<internal project user-agent>"
+UA = "hcp-effect-estimate-release"
 
```

## `hcpee/p29.py`  (from `phase2_9/p29.py`)

```diff
--- original/phase2_9/p29.py
+++ release/hcpee/p29.py
@@ -10,8 +10,7 @@
   * the representation builder (parcellated task-contrast COPE vectors, not spectra),
- [... 6 original lines removed here; see the release comment on the '+' side ...]
+  * the anatomical nuisance model (Part J).
+
+[release note] Vendored from the project's source tree. Only path/import wiring was
+changed and the unused twin estimators (historical Parts K/L/O) were removed; see
+docs/PATCHES.md. The estimator functions used by the scripts are unchanged.
 """
@@ -23,10 +22,4 @@
 HERE = os.path.dirname(os.path.abspath(__file__))
- [... 3 original lines removed here; see the release comment on the '+' side ...]
 sys.path.insert(0, HERE)
 import contrast as K                                                 # noqa: E402
- [... 3 original lines removed here; see the release comment on the '+' side ...]
 SEED = 20260823
@@ -231,161 +224,4 @@
 
- [... 159 original lines removed here; see the release comment on the '+' side ...]
+# [release] Parts K/L/O (monozygotic-twin estimators) removed: unused by this
+# study, which accessed no HCP Restricted Data and performed no twin analysis.
 
```

## `hcpee/contrast.py`  (from `phase2_5/contrast.py`)

```diff
--- original/phase2_5/contrast.py
+++ release/hcpee/contrast.py
@@ -24,18 +24,8 @@
 
-sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
-                                "..", "phase2"))
+sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 import analysis as A                                                 # noqa: E402
-import data as D                                                     # noqa: E402
+# [release] The EEG feature-table loader (`data as D`) and the two EEG-only helpers that
+# used it (rec_mean, deltas) are not shipped; see docs/PATCHES.md.
 
 METRICS = ("cosine", "correlation")
- [... 10 original lines removed here; see the release comment on the '+' side ...]
 
```

## `hcpee/p30.py`  (from `phase3_0/p30.py`)

```diff
--- original/phase3_0/p30.py
+++ release/hcpee/p30.py
@@ -14,5 +14,4 @@
 _HERE = os.path.dirname(os.path.abspath(__file__))
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+if _HERE not in sys.path:
+    sys.path.insert(0, _HERE)
 
@@ -22,6 +21,5 @@
 
- [... 4 original lines removed here; see the release comment on the '+' side ...]
+import paths                                                         # noqa: E402
+
+CACHE = paths.REST_CACHE          # parcellated resting-state timeseries
 
@@ -48,4 +46,3 @@
 def subjects():
-    p = os.path.join(RES29, "included_subjects.txt")
-    return [l.strip() for l in open(p) if l.strip()]
+    return [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]
 
@@ -53,3 +50,3 @@
 def atlas_labels():
-    return np.load(os.path.join(RES29, "atlas", "labels379.npy"))
+    return np.load(paths.ATLAS_LABELS)
 
@@ -124,9 +121,3 @@
 
- [... 8 original lines removed here; see the release comment on the '+' side ...]
+# [release] load_task() (reader of the historical z-statistic task cache) removed; the
+# effect-estimate scripts load COPE caches through hcpee.map_type_guard-checked loaders.
```

## `scripts/01_fetch_effect_estimates.py`  (from `phase2_9/02_fetch_effect_estimate.py`)

```diff
--- original/phase2_9/02_fetch_effect_estimate.py
+++ release/scripts/01_fetch_effect_estimates.py
@@ -8,2 +8,8 @@
 Maps are sourced from HCP Open Access bucket under the same permissions as z-stats.
+
+[release] Path/import wiring changed (hcpee/paths.py). Every object key is checked by
+hcpee.map_type_guard.assert_cope_key before it is requested, and a MAP_TYPE_MANIFEST.json
+declaring map_type=COPE_effect_estimate is written next to the cache; the analysis
+scripts refuse caches without it. Selection, parcellation and cache layout unchanged.
+See docs/PATCHES.md.
 """
@@ -17,3 +23,6 @@
 
-sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
+sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
+                                "hcpee"))
+import paths                                                          # noqa: E402
+import map_type_guard                                                 # noqa: E402
 import cifti
@@ -22,7 +31,5 @@
 
- [... 5 original lines removed here; see the release comment on the '+' side ...]
+RES = paths.OUT
+CACHE = paths.EE_CACHE
+KEYS = set()
 GRAYORDINATES = 91282
@@ -128,3 +135,3 @@
 
-def free_bytes(path="/"):
+def free_bytes(path=paths.EE_CACHE):
     st = os.statvfs(path)
@@ -136,2 +143,8 @@
     out_path = os.path.join(CACHE, f"{subj}_{ses}.npz")
+    for tname, tinfo in TRANSITION_COPE_NEEDS.items():          # [release] guard all keys
+        fd = level2_feat_dir(PREFIX[ses], subj, tinfo["task"])
+        for contrast in (tinfo["diff"], tinfo["B"], tinfo["A"]):
+            k = cope_key(fd, COPE_MAP[tinfo["task"]][contrast])
+            map_type_guard.assert_cope_key(k)
+            KEYS.add(k)
     if os.path.exists(out_path):
@@ -198,6 +211,6 @@
     # Load subject list (same 42 as Phase 2.9)
-    included = [l.strip() for l in open(os.path.join(RES29, "included_subjects.txt"))]
+    included = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]
     subs = included
 
-    labels = np.load(os.path.join(RES29, "atlas", "labels379.npy"))
+    labels = np.load(paths.ATLAS_LABELS)
 
@@ -215,2 +228,3 @@
                         key = cope_key(feat_dir, cope_idx)
+                        map_type_guard.assert_cope_key(key)
                         try:
@@ -237,2 +251,8 @@
         print(f"Exclusions: {len(exclusions)}")
+        if exclusions:
+            print("Manifest NOT written: exclusions present.", file=sys.stderr)
+            return 3
+        m = map_type_guard.write_manifest(CACHE, COPE_MAP, KEYS,
+                                          producer="scripts/01_fetch_effect_estimates.py")
+        print(f"Wrote {map_type_guard.MANIFEST_NAME} ({m['n_objects']} COPE objects).")
         return 0
```

## `scripts/02_fetch_rest.py`  (from `phase3_0/01_fetch_rest.py`)

```diff
--- original/phase3_0/01_fetch_rest.py
+++ release/scripts/02_fetch_rest.py
@@ -11,3 +11,9 @@
 
-Credentials come from <local credential path> via phase2_9.hcp_s3 and are never logged.
+Credentials come from ~/.hcp_aws.env (or $HCP_CRED_FILE) via hcpee.hcp_s3 and are
+never logged.
+
+[release] Path/import wiring changed (hcpee/paths.py): the cache goes to
+$HCPEE_REST_CACHE and the log to $HCPEE_OUT. Optional HCPEE_REST_ONLY
+("subject:session:run,..." ) restricts the job list, e.g. for spot checks.
+Parcellation and FC construction unchanged. See docs/PATCHES.md.
 """
@@ -23,3 +29,5 @@
 
-sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
+sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
+                                "hcpee"))
+import paths                                                          # noqa: E402
 import p30                                                            # noqa: E402
@@ -38,3 +46,3 @@
 
-def free_gb(path="/"):
+def free_gb(path=paths.REST_CACHE):
     st = shutil.disk_usage(path)
@@ -106,2 +114,6 @@
             for r in p30.ALL_RUNS]
+    only = os.environ.get("HCPEE_REST_ONLY")
+    if only:
+        want = {tuple(x.split(":")) for x in only.split(",") if x}
+        jobs = [j for j in jobs if j in want]
     state = dict(done=0, total=len(jobs), bytes=0)
@@ -125,3 +137,4 @@
         free_gb_after=round(free_gb(), 2), runs=recs)
-    p = os.path.join(p30.RES, "01_fetch_log.json")
+    os.makedirs(paths.OUT, exist_ok=True)
+    p = os.path.join(paths.OUT, "01_fetch_rest_log.json")
     json.dump(summary, open(p, "w"), indent=2)
```

## `scripts/10_interaction.py`  (from `phase3_1/01_interaction_effect_estimate.py`)

```diff
--- original/phase3_1/01_interaction_effect_estimate.py
+++ release/scripts/10_interaction.py
@@ -5,2 +5,6 @@
 (results/hcp_effect_estimate_falsification/cache/) instead of the z-stat cache.
+
+[release] Path/import wiring changed (hcpee/paths.py); the task cache is accepted only
+after hcpee.map_type_guard verifies it holds COPE effect estimates. Estimation code,
+seeds and permutation counts are unchanged. See docs/PATCHES.md.
 """
@@ -15,4 +19,5 @@
 HERE = os.path.dirname(os.path.abspath(__file__))
-sys.path.insert(0, HERE)
-sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase3_0"))
+sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hcpee"))
+import paths                                                          # noqa: E402
+import map_type_guard                                                 # noqa: E402
 import ix                                                             # noqa: E402
@@ -21,4 +26,5 @@
 
-ROOT = os.path.dirname(os.path.dirname(HERE))
-RES = os.path.join(ROOT, "results", "hcp_effect_estimate_falsification")
+RES = paths.OUT
+GUARD = {}
+PREREG = "HCP_EFFECT_ESTIMATE_FALSIFICATION_PREREG.md"
 T = list(P29.TRANSITIONS)
@@ -31,5 +37,3 @@
 def prereg_sha():
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+    return paths.prereg_sha(PREREG)
 
@@ -44,3 +48,4 @@
     data = {}
-    cache_dir = os.path.join(RES, "cache")
+    cache_dir = paths.EE_CACHE
+    GUARD.update(map_type_guard.assert_effect_estimate_cache(cache_dir, T, subs))
     for s in subs:
@@ -136,3 +141,5 @@
 
-    out = dict(prereg_sha256=prereg_sha(), seed=SEED, n_perm=NPERM, n_boot=NBOOT,
+    out = dict(prereg_sha256=prereg_sha(),
+               prereg_expected_sha256=paths.PREREG_EXPECTED_SHA256[PREREG],
+               map_type_guard=GUARD, seed=SEED, n_perm=NPERM, n_boot=NBOOT,
                transitions=T, n_subjects=len(subs), chance_rank1=1.0 / len(subs),
```

## `scripts/11_rest_control.py`  (from `phase3_1/02_rest_control_effect_estimate.py`)

```diff
--- original/phase3_1/02_rest_control_effect_estimate.py
+++ release/scripts/11_rest_control.py
@@ -6,2 +6,6 @@
 Everything else preserved exactly from canonical Phase 3.0.
+
+[release] Path/import wiring changed (hcpee/paths.py); the task cache is accepted only
+after hcpee.map_type_guard verifies it holds COPE effect estimates. Estimation code,
+seeds and permutation counts are unchanged. See docs/PATCHES.md.
 """
@@ -16,4 +20,5 @@
 HERE = os.path.dirname(os.path.abspath(__file__))
-sys.path.insert(0, HERE)
-sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase3_0"))
+sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hcpee"))
+import paths                                                          # noqa: E402
+import map_type_guard                                                 # noqa: E402
 import ix                                                             # noqa: E402
@@ -22,5 +27,5 @@
 
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+RES = paths.OUT
+GUARD = {}
+PREREG = "HCP_EFFECT_ESTIMATE_CONTROLS_PREREG.md"
 T = list(P29.TRANSITIONS)
@@ -34,5 +39,3 @@
 def prereg_sha():
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+    return paths.prereg_sha(PREREG)
 
@@ -114,3 +117,4 @@
     data = {}
-    cache_dir = os.path.join(EE_RES, "cache")
+    cache_dir = paths.EE_CACHE
+    GUARD.update(map_type_guard.assert_effect_estimate_cache(cache_dir, T, subs))
     for s in subs:
@@ -136,2 +140,4 @@
     out = {"prereg_sha256": prereg_sha(),
+           "prereg_expected_sha256": paths.PREREG_EXPECTED_SHA256[PREREG],
+           "map_type_guard": GUARD,
            "seed": SEED, "n_perm": NPERM, "k_ladder": list(K_LADDER),
```

## `scripts/12_anatomy_control.py`  (from `phase3_1/01_anatomy_interaction_effect_estimate.py`)

```diff
--- original/phase3_1/01_anatomy_interaction_effect_estimate.py
+++ release/scripts/12_anatomy_control.py
@@ -6,2 +6,6 @@
 Everything else preserved exactly from canonical Phase 3.2 (corrected modern anatomy model).
+
+[release] Path/import wiring changed (hcpee/paths.py); the task cache is accepted only
+after hcpee.map_type_guard verifies it holds COPE effect estimates. Estimation code,
+seeds and permutation counts are unchanged. See docs/PATCHES.md.
 """
@@ -16,5 +20,5 @@
 HERE = os.path.dirname(os.path.abspath(__file__))
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hcpee"))
+import paths                                                          # noqa: E402
+import map_type_guard                                                 # noqa: E402
 import an                                                             # noqa: E402
@@ -24,5 +28,5 @@
 
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+RES = paths.OUT
+GUARD = {}
+PREREG = "HCP_EFFECT_ESTIMATE_CONTROLS_PREREG.md"
 T = list(P29.TRANSITIONS)
@@ -39,5 +43,3 @@
 def prereg_sha():
- [... 3 original lines removed here; see the release comment on the '+' side ...]
+    return paths.prereg_sha(PREREG)
 
@@ -63,3 +65,3 @@
         sfx = "" if ses == "test" else "_retest"
-        z = np.load(os.path.join(ROOT, f"results/phase2_9/cache/anatomy{sfx}.npz"),
+        z = np.load(os.path.join(paths.ANAT_DIR, f"anatomy{sfx}.npz"),
                     allow_pickle=True)
@@ -215,3 +217,4 @@
     data = {}
-    cache_dir = os.path.join(EE_RES, "cache")
+    cache_dir = paths.EE_CACHE
+    GUARD.update(map_type_guard.assert_effect_estimate_cache(cache_dir, T, subs))
     for s in subs:
@@ -234,3 +237,5 @@
 
-    out = dict(prereg_sha256=prereg_sha(), seed=SEED, n_perm=NPERM, n_boot=NBOOT,
+    out = dict(prereg_sha256=prereg_sha(),
+               prereg_expected_sha256=paths.PREREG_EXPECTED_SHA256[PREREG],
+               map_type_guard=GUARD, seed=SEED, n_perm=NPERM, n_boot=NBOOT,
                b_shuf=BSHUF, k_primary=K_PRIMARY, lam_primary=LAM_PRIMARY,
```
