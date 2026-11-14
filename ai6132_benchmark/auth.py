"""Read private HF credentials without logging them or forwarding them to CDNs."""
import os
from pathlib import Path
import stat
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener


def read_hf_token(path=None):
    token = None
    if path is not None:
        credential = Path(path)
        info = credential.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('HF credential must be a current-user-owned regular file with mode 600')
        token = credential.read_text().strip()
    else:
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token and (not token.startswith('hf_') or any(char.isspace() for char in token)):
        raise ValueError('Invalid HF credential format')
    return token


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('HTTPS required')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(newurl).netloc != 'huggingface.co':
            redirected.remove_header('Authorization')
        return redirected


def hf_opener():
    return build_opener(SafeRedirect())
