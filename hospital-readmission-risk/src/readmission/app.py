"""Web page and prediction API: enter one encounter in a form, or upload a small CSV.

    python -m readmission.app                   # then open http://127.0.0.1:8000
    uvicorn --factory readmission.app:create_app

Privacy by design: requests are scored in memory, nothing is written to disk and no request data is logged
(the only log line names the kind of an unexpected error, never its message). Responses carry
only the row number, risk score, flag and explanatory text, never the submitted values. Error messages name
the problem field or row but do not repeat a value that was entered. The one exception is text entered where
a column heading, field name or medication name is expected and not recognized: it is shown back, shortened
to 40 characters, so it can be corrected (it is never stored or logged).

Two operating aids, neither of which records anything about a request:
- /api/stats shows running totals (encounters scored, how many were flagged, rejected inputs, unexpected
  errors) since this instance started. They are counts only and start again from zero on a restart.
- Notice mode: set the environment variable READMISSION_NOTICE to a sentence and the page shows it instead
  of the form, and scoring answers 503 with it. Unset the variable to return to normal.
"""

import argparse
import csv
import html
import io
import json
import logging
import os
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from readmission import __version__, config
from readmission.predict import DECLARATIONS, Predictor
from readmission.samples import build_sample_workbook
from readmission.schema import MAX_JSON_BYTES, MAX_ROWS, MAX_UPLOAD_BYTES, REQUIRED_COLUMNS

MAX_REPORTED_ERRORS = 20
MAX_NOTICE_CHARS = 500
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

logger = logging.getLogger("readmission")

NOTICE_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hospital Readmission Risk Prediction</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <header class="site-header">
    <h1>Hospital Readmission Risk Prediction</h1>
    <p class="tagline">AI/ML Study Project: from hospital data to a working prediction demo</p>
    <p class="notice" role="note">{notice}</p>
  </header>
</main>
</body>
</html>
"""


class UsageCounts:
    """Running totals since this instance started. Counts only: nothing about any single request is kept."""

    def __init__(self):
        self.since = datetime.now(UTC)
        self.requests_scored = 0
        self.encounters_scored = 0
        self.encounters_flagged = 0
        self.inputs_rejected = 0
        self.unexpected_errors = 0

    def scored(self, results):
        self.requests_scored += 1
        self.encounters_scored += len(results)
        self.encounters_flagged += sum(1 for result in results if result["flagged"])

    def summary(self):
        share = self.encounters_flagged / self.encounters_scored if self.encounters_scored else None
        return {
            "since_utc": self.since.isoformat(timespec="seconds"),
            "requests_scored": self.requests_scored,
            "encounters_scored": self.encounters_scored,
            "encounters_flagged": self.encounters_flagged,
            "flagged_share": None if share is None else round(share, 4),
            "inputs_rejected": self.inputs_rejected,
            "unexpected_errors": self.unexpected_errors,
        }


class InputError(Exception):
    def __init__(self, status, errors):
        super().__init__()
        self.status = status
        self.errors = errors[:MAX_REPORTED_ERRORS]


async def read_body(request, limit):
    """The request body, refusing anything larger than `limit` bytes."""
    too_large = InputError(413, [f"The request is too large (the limit is {limit // 1000} KB)."])
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise too_large
    return bytes(body)


def parse_csv(body):
    """Rows of an uploaded CSV as dicts, after checking size, header and row count."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(400, ["The file must be a UTF-8 text (CSV) file."]) from None
    try:
        return read_rows(text)
    except csv.Error:  # for example a spreadsheet (.xlsx) chosen instead of a CSV: not text the reader can split
        raise InputError(400, ["The file could not be read as a CSV. Save it as a comma-separated CSV file "
                               "(UTF-8) and try again."]) from None


def read_rows(text):
    reader = csv.DictReader(io.StringIO(text))
    header = [config.COLUMN_RENAMES.get(name, name) for name in (reader.fieldnames or [])]
    if not header:
        raise InputError(400, ["The file is empty. The first row must hold the column names."])
    problems = []
    if len(set(header)) != len(header):
        problems.append("Some column names appear more than once.")
    unknown = [name[:40] for name in header if name not in config.FEATURE_COLUMNS]
    if unknown:
        problems.append("Unknown columns: " + ", ".join(unknown) + ".")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing) + ".")
    if problems:
        raise InputError(422, problems)
    rows = list(reader)
    if not rows:
        raise InputError(422, ["The file has a header but no rows."])
    if len(rows) > MAX_ROWS:
        raise InputError(422, [f"The file has more than {MAX_ROWS} rows."])
    return rows


def render_declarations():
    """The declarations as one paragraph of text, with the first sentence of each in bold."""
    parts = []
    for text in DECLARATIONS:
        lead, separator, rest = text.partition(". ")
        bold, tail = (lead + ".", " " + rest) if separator else (text, "")
        parts.append(f"<strong>{html.escape(bold)}</strong>{html.escape(tail)}")
    return " ".join(parts)


def create_app(model_dir=None, notice=None):
    """The web app. `notice` (default: the READMISSION_NOTICE variable) switches on notice mode when non-empty."""
    predictor = Predictor.load(model_dir or os.environ.get("READMISSION_MODEL_DIR", "models"))
    notice = (os.environ.get("READMISSION_NOTICE", "") if notice is None else notice).strip()[:MAX_NOTICE_CHARS]
    counts = UsageCounts()
    web_dir = Path(str(resources.files("readmission") / "web"))
    # The declarations are written into the page itself, so every view shows them even before scripts run.
    page_html = (web_dir / "index.html").read_text(encoding="utf-8").replace(
        "{{DECLARATIONS}}", render_declarations())
    sample_workbook = build_sample_workbook(
        predictor.schema.guide(), (web_dir / "sample_encounters.csv").read_text(encoding="utf-8"))
    app = FastAPI(title="Hospital readmission risk", version=__version__,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        try:
            response = await call_next(request)
        except Exception as error:
            # Answered here, not in an exception handler: Starlette re-raises after a handler runs, and the
            # server would then log the full traceback, whose message comes from a library, not from our own
            # validation, and can contain submitted values.
            counts.unexpected_errors += 1
            logger.error("ERROR unexpected error while scoring: %s", type(error).__name__)
            response = JSONResponse({"errors": ["Something went wrong while scoring. Nothing was stored."]},
                                    status_code=500)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(InputError)
    async def input_error(request, error):
        counts.inputs_rejected += 1
        return JSONResponse({"errors": error.errors}, status_code=error.status)

    def respond(rows):
        scores = predictor.score(rows)
        results = [predictor.result(number, score) for number, score in enumerate(scores, start=1)]
        counts.scored(results)
        return {"results": results, "notes": predictor.notes(), "declarations": DECLARATIONS,
                "model": predictor.summary()}

    def paused():
        """The 503 answer for scoring requests while notice mode is on, otherwise None."""
        if notice:
            return JSONResponse({"errors": [notice]}, status_code=503)

    @app.get("/", include_in_schema=False)
    async def page():
        return HTMLResponse(NOTICE_PAGE.format(notice=html.escape(notice)) if notice else page_html)

    @app.get("/sample.csv", include_in_schema=False)
    async def sample():
        return FileResponse(web_dir / "sample_encounters.csv", media_type="text/csv",
                            filename="sample_encounters.csv")

    @app.get("/sample.xlsx", include_in_schema=False)
    async def sample_workbook_file():
        return Response(sample_workbook, media_type=XLSX_TYPE,
                        headers={"Content-Disposition": 'attachment; filename="sample_encounters.xlsx"'})

    @app.get("/api/health")
    async def health():
        # The versions let anyone confirm which release is live (useful after a deploy or a rollback).
        return {"status": "ok", **predictor.summary(), "app_version": __version__,
                "model_created_utc": predictor.metadata.get("created_at_utc"),
                "mode": "notice" if notice else "normal"}

    @app.get("/api/stats")
    async def stats():
        return counts.summary()

    @app.get("/api/schema")
    async def describe():
        return {**predictor.schema.describe(), "model": predictor.summary()}

    @app.post("/api/predict")
    async def predict(request: Request):
        if response := paused():
            return response
        try:
            payload = json.loads(await read_body(request, MAX_JSON_BYTES))
        except ValueError:
            raise InputError(400, ["The request was not valid JSON."]) from None
        if not isinstance(payload, dict):
            raise InputError(400, ["Expected a JSON object of field values."])
        medications = payload.pop("medications", {})
        if not isinstance(medications, dict):
            raise InputError(400, ["'medications' must be an object of drug name and change."])
        unknown = [str(name)[:40] for name in medications if name not in config.MEDICATION_COLUMNS]
        if unknown:
            raise InputError(422, ["Unknown medications: " + ", ".join(unknown) + "."])
        row, errors = predictor.schema.clean({**payload, **medications})
        if errors:
            raise InputError(422, errors)
        return respond([row])

    @app.post("/api/predict-file")
    async def predict_file(request: Request):
        if response := paused():
            return response
        rows, errors = [], []
        for number, raw in enumerate(parse_csv(await read_body(request, MAX_UPLOAD_BYTES)), start=1):
            if None in raw or None in raw.values():
                errors.append(f"Row {number}: the number of values does not match the header.")
                continue
            row, row_errors = predictor.schema.clean(raw)
            errors.extend(f"Row {number}, {message}" for message in row_errors)
            rows.append(row)
        if errors:
            raise InputError(422, errors)
        return respond(rows)

    app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")
    return app


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=None, help="folder holding the trained model (default: models)")
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
