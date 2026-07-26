"""
consolidar_examenes_serums.py

Consolida los CSV de exámenes SERUMS de múltiples convocatorias en una sola
base de datos maestra, agregando:
  - columna 'area'         : nombre del archivo CSV (sin extensión)
  - columna 'convocatoria' : nombre de la carpeta origen (ej. Examenes_2025_II)

Guarda la base maestra en tres formatos (Pickle, Parquet, CSV) y ejecuta una
batería de validaciones que se emiten al log y se persisten en JSON.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# El script se ancla al directorio donde vive el .py.
# Asume que las tres carpetas de convocatorias son hermanas del script.
BASE_DIR = Path(__file__).resolve().parent

FOLDERS = ["Examenes_2025_I", "Examenes_2025_II", "Examenes_2026_I"]
OUTPUT_DIR = BASE_DIR / "consolidado"
OUTPUT_NAME = "examenes_serums_master"
INPUT_ENCODING = "utf-8-sig"

# Estructura esperada de cada CSV de examen
EXPECTED_COLUMNS = [
    "questions",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "correct_answer",
]
VALID_ANSWERS = {"A", "B", "C", "D"}
CRITICAL_COLUMNS = ["questions", "option_A", "option_B", "option_C", "option_D"]

# Cuántos caracteres de la pregunta mostrar en el log de duplicados
DUPLICATE_PREVIEW_CHARS = 120

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("consolidar_serums")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Estructuras de tracking y validación
# ---------------------------------------------------------------------------

@dataclass
class FileTracking:
    convocatoria: str
    area: str
    path: str
    rows_read: int
    columns: list[str]
    read_ok: bool
    error: str | None = None


@dataclass
class ValidationReport:
    timestamp: str
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_error(self, check: str, message: str, **details) -> None:
        self.errors.append({"check": check, "message": message, **details})

    def add_warning(self, check: str, message: str, **details) -> None:
        self.warnings.append({"check": check, "message": message, **details})

    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lectura de CSV individual
# ---------------------------------------------------------------------------

def read_csv_with_metadata(
    csv_path: Path,
    convocatoria: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame | None, FileTracking]:
    area = csv_path.stem

    try:
        df = pd.read_csv(
            csv_path,
            encoding=INPUT_ENCODING,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as e:
        logger.error(f"  ✗ Error leyendo {csv_path.name}: {e}")
        tracking = FileTracking(
            convocatoria=convocatoria,
            area=area,
            path=str(csv_path),
            rows_read=0,
            columns=[],
            read_ok=False,
            error=str(e),
        )
        return None, tracking

    tracking = FileTracking(
        convocatoria=convocatoria,
        area=area,
        path=str(csv_path),
        rows_read=len(df),
        columns=list(df.columns),
        read_ok=True,
    )

    if df.empty:
        logger.warning(f"  ⚠ Archivo vacío: {csv_path.name}")
        return None, tracking

    df["area"] = area
    df["convocatoria"] = convocatoria

    logger.info(f"  ✓ {csv_path.name:<50} | filas: {len(df):>4} | area='{area}'")
    return df, tracking


# ---------------------------------------------------------------------------
# Recorrido de carpetas
# ---------------------------------------------------------------------------

def collect_csvs_from_folder(
    folder_path: Path,
    logger: logging.Logger,
) -> tuple[list[pd.DataFrame], list[FileTracking]]:
    convocatoria = folder_path.name
    trackings: list[FileTracking] = []
    dataframes: list[pd.DataFrame] = []

    if not folder_path.exists():
        logger.error(f"Carpeta no existe: {folder_path}")
        return dataframes, trackings

    csv_files = sorted(folder_path.glob("*.csv"))
    logger.info(f"[{convocatoria}] — {len(csv_files)} archivo(s) CSV encontrado(s)")

    for csv_path in csv_files:
        df, tracking = read_csv_with_metadata(csv_path, convocatoria, logger)
        trackings.append(tracking)
        if df is not None:
            dataframes.append(df)

    return dataframes, trackings


def consolidate_exams(
    base_dir: Path,
    folders: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[FileTracking]]:
    all_frames: list[pd.DataFrame] = []
    all_trackings: list[FileTracking] = []

    for folder_name in folders:
        folder_path = base_dir / folder_name
        frames, trackings = collect_csvs_from_folder(folder_path, logger)
        all_frames.extend(frames)
        all_trackings.extend(trackings)

    if not all_frames:
        raise RuntimeError("No se pudo leer ningún CSV. Revisa las rutas.")

    master = pd.concat(all_frames, ignore_index=True, sort=False)

    meta_cols = ["convocatoria", "area"]
    other_cols = [c for c in master.columns if c not in meta_cols]
    master = master[meta_cols + other_cols]

    return master, all_trackings


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------

def validate_row_counts(
    df: pd.DataFrame,
    trackings: list[FileTracking],
    report: ValidationReport,
    logger: logging.Logger,
) -> None:
    check = "row_counts_reconciliation"
    report.checks_run.append(check)
    logger.info(f"[validación] {check} ...")

    expected: dict[tuple[str, str], int] = {}
    for t in trackings:
        if not t.read_ok:
            continue
        key = (t.convocatoria, t.area)
        expected[key] = expected.get(key, 0) + t.rows_read

    actual_series = df.groupby(["convocatoria", "area"], sort=False).size()
    actual: dict[tuple[str, str], int] = {
        (conv, area): int(n) for (conv, area), n in actual_series.items()
    }

    all_keys = set(expected) | set(actual)
    mismatches = 0
    for key in sorted(all_keys):
        exp = expected.get(key, 0)
        act = actual.get(key, 0)
        if exp != act:
            mismatches += 1
            report.add_error(
                check,
                f"Discrepancia en {key}: leídas={exp}, en maestro={act}",
                convocatoria=key[0],
                area=key[1],
                rows_read=exp,
                rows_in_master=act,
                diff=act - exp,
            )
            logger.error(
                f"  ✗ {key[0]} / {key[1]}: leídas={exp}, maestro={act} "
                f"(diff={act - exp:+d})"
            )

    if mismatches == 0:
        logger.info(f"  ✓ Sin discrepancias en {len(all_keys)} grupo(s) "
                    f"(total filas: {sum(expected.values())})")


def validate_columns(
    trackings: list[FileTracking],
    report: ValidationReport,
    logger: logging.Logger,
) -> None:
    check = "expected_columns"
    report.checks_run.append(check)
    logger.info(f"[validación] {check} ...")

    expected_set = set(EXPECTED_COLUMNS)
    problems = 0

    for t in trackings:
        if not t.read_ok:
            continue
        got_set = set(t.columns)
        missing = expected_set - got_set
        extra = got_set - expected_set

        if missing or extra:
            problems += 1
            report.add_error(
                check,
                f"Columnas inesperadas en {t.convocatoria}/{t.area}",
                convocatoria=t.convocatoria,
                area=t.area,
                missing=sorted(missing),
                extra=sorted(extra),
                got=t.columns,
            )
            logger.error(
                f"  ✗ {t.convocatoria}/{t.area}: "
                f"faltan={sorted(missing)}, sobran={sorted(extra)}"
            )

    if problems == 0:
        logger.info(f"  ✓ Estructura correcta en {len(trackings)} archivo(s)")


def validate_answer_column(
    df: pd.DataFrame,
    report: ValidationReport,
    logger: logging.Logger,
) -> None:
    check = "correct_answer_values"
    report.checks_run.append(check)
    logger.info(f"[validación] {check} ...")

    if "correct_answer" not in df.columns:
        report.add_error(check, "Columna 'correct_answer' no existe en el maestro")
        logger.error("  ✗ Columna 'correct_answer' no existe")
        return

    normalized = df["correct_answer"].astype(str).str.strip().str.upper()
    invalid_mask = ~normalized.isin(VALID_ANSWERS)
    n_invalid = int(invalid_mask.sum())

    if n_invalid == 0:
        logger.info(f"  ✓ Todas las respuestas están en {sorted(VALID_ANSWERS)}")
        return

    bad_values = df.loc[invalid_mask, "correct_answer"].value_counts().to_dict()
    per_area = (
        df.loc[invalid_mask]
        .groupby(["convocatoria", "area"])
        .size()
        .to_dict()
    )

    report.add_warning(
        check,
        f"{n_invalid} fila(s) con correct_answer fuera de {sorted(VALID_ANSWERS)}",
        count=n_invalid,
        unique_bad_values={str(k): int(v) for k, v in bad_values.items()},
        by_area={f"{k[0]}|{k[1]}": int(v) for k, v in per_area.items()},
    )
    logger.warning(f"  ⚠ {n_invalid} respuesta(s) inválida(s). "
                   f"Valores encontrados: {bad_values}")


def validate_empty_cells(
    df: pd.DataFrame,
    report: ValidationReport,
    logger: logging.Logger,
) -> None:
    check = "empty_critical_cells"
    report.checks_run.append(check)
    logger.info(f"[validación] {check} ...")

    total_empty = 0
    per_col: dict[str, int] = {}

    for col in CRITICAL_COLUMNS:
        if col not in df.columns:
            continue
        empty_mask = df[col].astype(str).str.strip() == ""
        n = int(empty_mask.sum())
        per_col[col] = n
        total_empty += n

    if total_empty == 0:
        logger.info(f"  ✓ Sin celdas vacías en columnas críticas")
        return

    report.add_warning(
        check,
        f"{total_empty} celda(s) vacía(s) en columnas críticas",
        by_column=per_col,
    )
    for col, n in per_col.items():
        if n > 0:
            logger.warning(f"  ⚠ Columna '{col}': {n} celda(s) vacía(s)")


def _truncate(text: str, n: int = DUPLICATE_PREVIEW_CHARS) -> str:
    """Trunca texto para vista previa en consola, colapsando saltos de línea."""
    flat = " ".join(text.split())
    if len(flat) <= n:
        return flat
    return flat[:n].rstrip() + "…"


def validate_duplicates(
    df: pd.DataFrame,
    report: ValidationReport,
    logger: logging.Logger,
) -> None:
    """
    Detecta preguntas duplicadas dentro del mismo (convocatoria, area) e imprime
    el texto de cada pregunta duplicada + las filas donde aparece.

    Duplicados entre áreas o convocatorias distintas NO se reportan (esperable).
    """
    check = "duplicate_questions_within_area"
    report.checks_run.append(check)
    logger.info(f"[validación] {check} ...")

    if "questions" not in df.columns:
        logger.warning("  ⚠ No hay columna 'questions', se omite chequeo")
        return

    # Normalización para comparar (espacios colapsados, minúsculas)
    normalized = (
        df["questions"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    work = df[["convocatoria", "area", "questions"]].copy()
    work["_q_norm"] = normalized
    work["_row_idx"] = df.index  # índice original en el maestro

    # Un grupo duplicado = misma (convocatoria, area, _q_norm) con >1 fila
    grouped = work.groupby(
        ["convocatoria", "area", "_q_norm"], sort=False
    )
    dup_groups = [g for _, g in grouped if len(g) > 1]

    if not dup_groups:
        logger.info(f"  ✓ Sin preguntas duplicadas dentro de cada área")
        return

    n_rows_involved = sum(len(g) for g in dup_groups)
    n_groups = len(dup_groups)

    logger.warning(
        f"  ⚠ {n_groups} grupo(s) de duplicados "
        f"({n_rows_involved} fila(s) involucrada(s))"
    )

    # Detalle por grupo — al log + al reporte
    duplicate_details: list[dict] = []
    counts_per_area: dict[tuple[str, str], int] = {}

    for g in dup_groups:
        conv = g["convocatoria"].iloc[0]
        area = g["area"].iloc[0]
        # El texto original de la primera ocurrencia (sin normalizar)
        original_text = g["questions"].iloc[0]
        occurrences = len(g)
        row_indices = [int(i) for i in g["_row_idx"].tolist()]

        counts_per_area[(conv, area)] = counts_per_area.get((conv, area), 0) + occurrences

        # Log legible
        logger.warning(
            f"    • {conv}/{area} — {occurrences} ocurrencia(s) "
            f"en filas {row_indices}"
        )
        logger.warning(f"      \"{_truncate(original_text)}\"")

        # Detalle completo al JSON
        duplicate_details.append({
            "convocatoria": conv,
            "area": area,
            "occurrences": occurrences,
            "row_indices": row_indices,
            "question_text": original_text,
        })

    report.add_warning(
        check,
        f"{n_rows_involved} fila(s) involucrada(s) en {n_groups} grupo(s) de duplicados",
        n_groups=n_groups,
        n_rows_involved=n_rows_involved,
        by_area={f"{k[0]}|{k[1]}": int(v) for k, v in counts_per_area.items()},
        duplicates=duplicate_details,
    )


def run_validations(
    df: pd.DataFrame,
    trackings: list[FileTracking],
    logger: logging.Logger,
) -> ValidationReport:
    report = ValidationReport(timestamp=datetime.now().isoformat())

    logger.info("═" * 70)
    logger.info("Iniciando validaciones")
    logger.info("═" * 70)

    validate_columns(trackings, report, logger)
    validate_row_counts(df, trackings, report, logger)
    validate_answer_column(df, report, logger)
    validate_empty_cells(df, report, logger)
    validate_duplicates(df, report, logger)

    report.summary = {
        "total_rows": int(len(df)),
        "total_files_read_ok": sum(1 for t in trackings if t.read_ok),
        "total_files_failed": sum(1 for t in trackings if not t.read_ok),
        "n_errors": len(report.errors),
        "n_warnings": len(report.warnings),
    }

    logger.info("─" * 70)
    logger.info(f"Validaciones: {len(report.errors)} error(es), "
                f"{len(report.warnings)} advertencia(s)")
    logger.info("─" * 70)

    return report


def save_validation_report(
    report: ValidationReport,
    trackings: list[FileTracking],
    output_dir: Path,
    output_name: str,
    logger: logging.Logger,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{output_name}_validation.json"

    payload = {
        "report": report.to_dict(),
        "files_tracking": [asdict(t) for t in trackings],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Reporte de validación: {path}")
    return path


# ---------------------------------------------------------------------------
# Guardado de la base maestra
# ---------------------------------------------------------------------------

def save_master_database(
    df: pd.DataFrame,
    output_dir: Path,
    output_name: str,
    logger: logging.Logger,
) -> dict[str, Path]:
    """
    Guarda pickle, parquet y CSV.

    NOTA: Excel puede convertir strings como '10-3' a fechas al abrir CSV con
    doble-click aún con QUOTE_ALL. Para análisis usa .parquet o .pkl, o abre el
    CSV en Excel vía Datos > Desde texto/CSV marcando columnas como Texto.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / f"{output_name}.pkl"
    parquet_path = output_dir / f"{output_name}.parquet"
    csv_path = output_dir / f"{output_name}.csv"

    df.to_pickle(pkl_path)
    logger.info(f"✓ Pickle guardado : {pkl_path}")

    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"✓ Parquet guardado: {parquet_path}")

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    logger.info(f"✓ CSV guardado   : {csv_path}")

    return {"pickle": pkl_path, "parquet": parquet_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, logger: logging.Logger) -> None:
    logger.info("─" * 70)
    logger.info(f"Total de filas    : {len(df):,}")
    logger.info(f"Total de columnas : {len(df.columns)}")
    logger.info(f"Columnas          : {list(df.columns)}")
    logger.info("Distribución por convocatoria:")
    for conv, n in df["convocatoria"].value_counts().sort_index().items():
        logger.info(f"  • {conv:<25} {n:>5} preguntas")
    logger.info("Distribución por área (top 20):")
    for area, n in df["area"].value_counts().head(20).items():
        logger.info(f"  • {area:<40} {n:>5} preguntas")
    logger.info("─" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validación temprana antes de tocar logging
    if not BASE_DIR.exists():
        print(f"ERROR: BASE_DIR no existe: {BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    missing = [f for f in FOLDERS if not (BASE_DIR / f).exists()]
    if missing:
        print(f"ERROR: faltan carpetas dentro de {BASE_DIR}: {missing}",
              file=sys.stderr)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"consolidacion_{stamp}.log"
    logger = setup_logging(log_path)

    logger.info("═" * 70)
    logger.info("Consolidación de exámenes SERUMS")
    logger.info(f"Directorio base : {BASE_DIR}")
    logger.info(f"Carpetas        : {FOLDERS}")
    logger.info(f"Salida          : {OUTPUT_DIR}")
    logger.info("═" * 70)

    # 1) Consolidar
    try:
        master, trackings = consolidate_exams(BASE_DIR, FOLDERS, logger)
    except Exception as e:
        logger.exception(f"Error fatal durante la consolidación: {e}")
        sys.exit(1)

    summarize(master, logger)

    # 2) Validar (no rompe pipeline; acumula hallazgos)
    report = run_validations(master, trackings, logger)
    save_validation_report(report, trackings, OUTPUT_DIR, OUTPUT_NAME, logger)

    # 3) Guardar base maestra
    try:
        save_master_database(master, OUTPUT_DIR, OUTPUT_NAME, logger)
    except Exception as e:
        logger.exception(f"Error al guardar la base maestra: {e}")
        sys.exit(1)

    # 4) Estado final
    if report.errors:
        logger.warning(
            f"⚠ Proceso completado CON {len(report.errors)} error(es) "
            f"de validación. Revisa el JSON de reporte."
        )
        sys.exit(2)
    elif report.warnings:
        logger.info(
            f"✓ Proceso completado con {len(report.warnings)} advertencia(s). "
            f"Revisa el JSON de reporte."
        )
    else:
        logger.info("✓ Proceso completado sin errores ni advertencias.")


if __name__ == "__main__":
    main()