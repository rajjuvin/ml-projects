"""Web page and prediction API: enter one encounter in a form, or upload a small CSV.

    python -m readmission.app                   # then open http://127.0.0.1:8000
    uvicorn --factory readmission.app:create_app

Privacy by design: requests are scored in memory and nothing is written to disk or logged. Responses carry
only the row number, risk score, flag and explanatory text, never the submitted values, and error messages
name the problem column but never repeat what was entered.
"""

import argparse
import csv
import html
import io
import json
import os
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
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


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


def create_app(model_dir=None):
    predictor = Predictor.load(model_dir or os.environ.get("READMISSION_MODEL_DIR", "models"))
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
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(InputError)
    async def input_error(request, error):
        return JSONResponse({"errors": error.errors}, status_code=error.status)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse({"errors": ["Something went wrong while scoring. Nothing was stored."]},
                            status_code=500)

    def respond(rows):
        scores = predictor.score(rows)
        return {"results": [predictor.result(number, score) for number, score in enumerate(scores, start=1)],
                "notes": predictor.notes(), "declarations": DECLARATIONS, "model": predictor.summary()}

    @app.get("/", include_in_schema=False)
    async def page():
        return HTMLResponse(page_html)

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
        return {"status": "ok", **predictor.summary()}

    @app.get("/api/schema")
    async def describe():
        return {**predictor.schema.describe(), "model": predictor.summary()}

    @app.post("/api/predict")
    async def predict(request: Request):
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
