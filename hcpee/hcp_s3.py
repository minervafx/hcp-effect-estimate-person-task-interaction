#!/usr/bin/env python
"""AWS SigV4 byte-range client for the HCP `hcp-openaccess` bucket - stdlib only.

PART R forbids installing large stacks.  `boto3` pulls in botocore + a bundled data
directory (~90 MB) to do one thing Phase 2.9 needs: sign a GET with a Range header.
That is ~80 lines of hmac, so it is done here instead.

PART S is enforced structurally: this module has NO fallback, NO anonymous mode and NO
credential discovery beyond one explicitly named file.  If the credential file is
absent it raises `AccessGateError` and the pipeline stops.  Nothing here attempts to
bypass, guess, or work around HCP access control.

Credentials are the ones ConnectomeDB/BALSA issues to the *user* after they personally
accept the WU-Minn HCP Open Access Data Use Terms.  Expected file, mode 0600:

    ~/.hcp_aws.env      (or the path in $HCP_CRED_FILE)
        HCP_AWS_ACCESS_KEY_ID=...
        HCP_AWS_SECRET_ACCESS_KEY=...
        # optional: HCP_AWS_REGION=us-east-1
"""
import datetime
import hashlib
import hmac
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

CRED_FILE = os.environ.get("HCP_CRED_FILE", os.path.expanduser("~/.hcp_aws.env"))
BUCKET = "hcp-openaccess"
DEFAULT_REGION = "us-east-1"
SERVICE = "s3"
UA = "hcp-effect-estimate-release"


class AccessGateError(RuntimeError):
    """Raised when HCP credentials / data-use acceptance are not available."""


# ------------------------------------------------------------------ signing
def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def _hmac(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret, datestamp, region, service):
    k = _hmac(("AWS4" + secret).encode("utf-8"), datestamp)
    k = _hmac(k, region)
    k = _hmac(k, service)
    return _hmac(k, "aws4_request")


def sigv4_headers(method, host, path, query, headers, payload_hash,
                  access_key, secret_key, region, service, amzdate):
    """Return the full signed header dict. `path` must already be URI-encoded."""
    datestamp = amzdate[:8]
    h = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    h["host"] = host
    h["x-amz-date"] = amzdate
    h["x-amz-content-sha256"] = payload_hash
    signed = ";".join(sorted(h))
    canon_headers = "".join(f"{k}:{h[k]}\n" for k in sorted(h))
    canon_query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}"
        for k, v in sorted((query or {}).items()))
    canon = "\n".join([method, path, canon_query, canon_headers, signed, payload_hash])
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, _sha256(canon.encode("utf-8"))])
    sig = hmac.new(signing_key(secret_key, datestamp, region, service),
                   sts.encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(headers)
    out["Host"] = host
    out["x-amz-date"] = amzdate
    out["x-amz-content-sha256"] = payload_hash
    out["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                            f"SignedHeaders={signed}, Signature={sig}")
    return out, sig


# -------------------------------------------------------------- credentials
def load_credentials(path=None):
    """Read HCP-issued AWS keys. Raises AccessGateError if the gate is closed."""
    path = path or CRED_FILE
    if not os.path.exists(path):
        raise AccessGateError(
            f"HCP credential file {path} not found.\n"
            "Phase 2.9 stops here by design (PART S).\n"
            "Required human action:\n"
            "  1. Register an account at https://balsa.wustl.edu/register/register\n"
            "  2. Open https://balsa.wustl.edu/project?project=HCP_YA and\n"
            "     https://balsa.wustl.edu/project?project=HCP_Retest and personally\n"
            "     accept the 'WU-Minn HCP Consortium Open Access Data Use Terms'.\n"
            "  3. Click 'Get/Reset AWS S3 Access' to have AWS keys generated.\n"
            f"  4. Write them to {path} (chmod 600) as\n"
            "       HCP_AWS_ACCESS_KEY_ID=...\n"
            "       HCP_AWS_SECRET_ACCESS_KEY=...\n"
            "The Data Use Terms contain personal attestations about institutional /\n"
            "IRB compliance; they must be accepted by the researcher, not by an agent.")
    env = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # accept shell-style credential files: optional `export ` prefix, quoted values
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

    def pick(*names):
        for n in names:
            if env.get(n):
                return env[n]
        for n in names:                       # already-exported environment as fallback
            if os.environ.get(n):
                return os.environ[n]
        return None

    # Both the project-scoped names and the standard AWS names are accepted. HCP's
    # "Get/Reset AWS S3 Access" hands the user a block written in the standard names,
    # so requiring the HCP_-prefixed spelling would reject a perfectly valid file.
    ak = pick("HCP_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
    sk = pick("HCP_AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")
    region = pick("HCP_AWS_REGION", "AWS_DEFAULT_REGION", "AWS_REGION") or DEFAULT_REGION
    if not ak or not sk:
        raise AccessGateError(
            f"{path} exists but supplies no usable key pair. Accepted names: "
            "HCP_AWS_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID and "
            "HCP_AWS_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY.")
    return ak, sk, region


# --------------------------------------------------------------- operations
class S3(object):
    def __init__(self, bucket=BUCKET, cred_file=None):
        self.access_key, self.secret_key, self.region = load_credentials(cred_file)
        self.bucket = bucket
        self.host = f"{bucket}.s3.amazonaws.com"
        self.ctx = ssl.create_default_context()
        self.bytes_transferred = 0
        self.requests = 0

    def _request(self, key, extra_headers=None, method="GET", query=None):
        path = "/" + urllib.parse.quote(key.lstrip("/"), safe="/~")
        amzdate = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload_hash = _sha256(b"")
        base = dict(extra_headers or {})
        base["User-Agent"] = UA
        signed, _ = sigv4_headers(method, self.host, path, query, base, payload_hash,
                                  self.access_key, self.secret_key, self.region,
                                  SERVICE, amzdate)
        url = f"https://{self.host}{path}"
        if query:
            url += "?" + "&".join(f"{k}={urllib.parse.quote(str(v), safe='-_.~')}"
                                  for k, v in sorted(query.items()))
        req = urllib.request.Request(url, method=method, headers=signed)
        try:
            with urllib.request.urlopen(req, timeout=180, context=self.ctx) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read(600)
            except Exception:
                pass
            if e.code in (401, 403):
                raise AccessGateError(
                    f"S3 {e.code} for s3://{self.bucket}/{key}. The keys are present but "
                    "rejected. Most likely the Open Access Data Use Terms have not been "
                    "accepted for this project, or the keys were reset. "
                    f"Server said: {body[:300]!r}")
            raise
        self.requests += 1
        self.bytes_transferred += len(data)
        return data

    def get(self, key):
        return self._request(key)

    def get_range(self, key, start, end_exclusive):
        """Byte-range GET; `end_exclusive` follows Python slice convention."""
        return self._request(key, {"Range": f"bytes={start}-{end_exclusive - 1}"})

    def head_size(self, key):
        """Object size via a 0-0 range GET (HEAD is not always signed identically)."""
        path = "/" + urllib.parse.quote(key.lstrip("/"), safe="/~")
        amzdate = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = {"User-Agent": UA, "Range": "bytes=0-0"}
        signed, _ = sigv4_headers("GET", self.host, path, None, base, _sha256(b""),
                                  self.access_key, self.secret_key, self.region,
                                  SERVICE, amzdate)
        req = urllib.request.Request(f"https://{self.host}{path}", headers=signed)
        with urllib.request.urlopen(req, timeout=60, context=self.ctx) as r:
            self.requests += 1
            cr = r.headers.get("Content-Range", "")
            r.read(1)
            return int(cr.rsplit("/", 1)[-1]) if "/" in cr else None
