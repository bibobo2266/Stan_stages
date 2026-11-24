#!/usr/bin/env python3
"""Send an HTML file as the mail body (UTF-8, no markdown conversion). One copy only.
Env: GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO. Args: --file --subject [--from-name]."""
import argparse, os, smtplib
from email.message import EmailMessage
from email.utils import formataddr
ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--subject", required=True)
ap.add_argument("--from-name", default="")
ap.add_argument("--attach", nargs="*", default=[])
a = ap.parse_args()
html = open(a.file, encoding="utf-8").read()
if "<html" not in html.lower():
    html = "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>"
user = os.environ["GMAIL_USER"]
to = os.environ["MAIL_TO"].strip()
if not to:
    raise SystemExit("NO_RECIPIENT")
m = EmailMessage()
m["Subject"] = a.subject
m["From"] = formataddr((a.from_name, user)) if a.from_name else user
m["To"] = to
m.set_content("請用支援 HTML 的信箱閱讀這封信。")
m.add_alternative(html, subtype="html", charset="utf-8")
for f in a.attach:
    if os.path.isfile(f):
        m.add_attachment(open(f, "rb").read(), maintype="text", subtype="csv", filename=os.path.basename(f))
with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
    s.login(user, os.environ["GMAIL_APP_PASSWORD"])
    s.send_message(m)
print("SENT_HTML_MAIL", a.subject)
